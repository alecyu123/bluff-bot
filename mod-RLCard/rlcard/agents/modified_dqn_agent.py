import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy

# Keep rlcard import if you are using it, otherwise remove if standalone
# from rlcard.utils.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])


class VAEDQNAgent(object):
    '''
    DQN Agent using an LSTM-VAE Estimator network.
    Modified to handle LSTM-VAE parameters and checkpointing.
    '''
    def __init__(self,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 num_actions=2,
                 state_shape=None,
                 train_every=1,
                 # ADD LSTM-VAE parameters
                 lstm_input_size=None,
                 lstm_hidden_size=128,
                 lstm_num_layers=1,
                 latent_dim=32,
                 kld_weight=0.001, # Added KLD weight param here
                 learning_rate=0.00005,
                 device=None,
                 save_path=None,
                 save_every=float('inf'),):

        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.total_t = 0
        self.train_t = 0
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # Create estimators using LSTM-VAE parameters
        estimator_args = {
            'num_actions': num_actions,
            'learning_rate': learning_rate,
            'state_shape': state_shape,
            'device': self.device,
            'lstm_input_size': lstm_input_size,
            'lstm_hidden_size': lstm_hidden_size,
            'lstm_num_layers': lstm_num_layers,
            'latent_dim': latent_dim,
            'kld_weight': kld_weight
        }
        self.q_estimator = Estimator(**estimator_args)
        self.target_estimator = Estimator(**estimator_args)

        # Synchronize target network weights initially
        self.target_estimator.qnet.load_state_dict(self.q_estimator.qnet.state_dict())
        self.target_estimator.qnet.eval()

        self.memory = Memory(replay_memory_size, batch_size)
    
        self.save_path = save_path
        self.save_every = save_every

    def feed(self, ts):
        ''' Store data in to replay buffer and train the agent. '''
        (state, action, reward, next_state, done) = tuple(ts)
        current_obs = state['obs']
        next_obs = next_state['obs']
        legal_actions_keys = list(next_state['legal_actions'].keys())

        self.feed_memory(current_obs, action, reward, next_obs, legal_actions_keys, done)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp>=0 and tmp%self.train_every == 0:
            self.train()

    def step(self, state):
        ''' Predict the action using epsilon-greedy policy. '''
        q_values = self.predict(state)
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        legal_actions = list(state['legal_actions'].keys())

        if not legal_actions:
            return 0

        if random.random() < epsilon:
            action = random.choice(legal_actions)
        else:
             q_subset = q_values[legal_actions]
             if np.all(np.isinf(q_subset)):
                 action = random.choice(legal_actions)
             else:
                 action = legal_actions[np.argmax(q_subset)]

        return action

    def eval_step(self, state):
        ''' Predict the action for evaluation purpose. '''
        q_values = self.predict(state)
        legal_actions = list(state['legal_actions'].keys())

        if not legal_actions:
             best_action = 0
             info = {'values': {}}
             return best_action, info

        q_subset = q_values[legal_actions]
        if np.all(np.isinf(q_subset)):
             best_action = random.choice(legal_actions)
        else:
            best_action = legal_actions[np.argmax(q_subset)]

        info = {}
        raw_legal_actions = state.get('raw_legal_actions', legal_actions)
        values_dict = {}
        for i, action_key in enumerate(legal_actions):
             # This assumes raw_legal_actions[i] corresponds to legal_actions[i]
             # Ensure this correspondence holds in your environment data
             raw_key = raw_legal_actions[i]
             q_val = q_values[action_key]
             values_dict[raw_key] = float(q_val) if np.isfinite(q_val) else -float('inf')
        info['values'] = values_dict

        return best_action, info

    def predict(self, state):
        ''' Predict the masked Q-values '''
        obs = np.array(state['obs'])
        q_values = self.q_estimator.predict_nograd(np.expand_dims(obs, 0))[0]

        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
        legal_actions = list(state['legal_actions'].keys())
        valid_legal_actions = [a for a in legal_actions if 0 <= a < self.num_actions]

        if valid_legal_actions:
            masked_q_values[valid_legal_actions] = q_values[valid_legal_actions]

        return masked_q_values

    def train(self):
        ''' Train the network. '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()

        # Calculate best next actions using Q-network (Double DQN - Step 1: Action Selection)
        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        best_actions = np.zeros(self.batch_size, dtype=int)
        for i in range(self.batch_size):
             legal_acts = legal_actions_batch[i]
             if not legal_acts:
                  best_actions[i] = 0
                  continue
             valid_legal_acts = [a for a in legal_acts if 0 <= a < self.num_actions]
             if not valid_legal_acts:
                  best_actions[i] = 0
                  continue
             q_subset = q_values_next[i, valid_legal_acts]
             if np.all(np.isinf(q_subset)):
                  best_actions[i] = random.choice(valid_legal_acts)
             else:
                  argmax_local = np.argmax(q_subset)
                  best_actions[i] = valid_legal_acts[argmax_local]


        # Evaluate best next actions using Target-network (Double DQN - Step 2: Action Evaluation)
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        target_q_for_best_action = q_values_next_target[np.arange(self.batch_size), best_actions]

        # Calculate TD Target (y)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            self.discount_factor * target_q_for_best_action

        # Perform gradient descent update
        state_batch = np.array(state_batch)
        loss = self.q_estimator.update(state_batch, action_batch, target_batch)

        print('\rINFO - Step {}, rl-loss: {:.4f}'.format(self.total_t, loss), end='')

        # Update the target estimator using load_state_dict
        if self.train_t > 0 and self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator.qnet.load_state_dict(self.q_estimator.qnet.state_dict())
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1

        # Save checkpoint logic remains the same externally
        if self.save_path and self.train_t % self.save_every == 0:
            self.save_checkpoint(self.save_path)
            print("\nINFO - Saved model checkpoint.")

        return loss

    def feed_memory(self, state, action, reward, next_state, legal_actions, done):
        ''' Feed transition to memory '''
        self.memory.save(state, action, reward, next_state, legal_actions, done)

    def set_device(self, device):
        ''' Set the device for the agent and its estimators. '''
        self.device = device
        self.q_estimator.device = device
        self.q_estimator.qnet.to(device)
        self.target_estimator.device = device
        self.target_estimator.qnet.to(device)

    def checkpoint_attributes(self):
        ''' Return the current checkpoint attributes (dict). '''
        q_estimator_attrs = self.q_estimator.checkpoint_attributes()

        current_epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        epsilon_start_val = self.epsilons[0]
        epsilon_end_val = self.epsilons[-1]

        return {
            'agent_type': 'VAEDQNAgent',
            'q_estimator': q_estimator_attrs,
            'memory': self.memory.checkpoint_attributes(),
            'total_t': self.total_t,
            'train_t': self.train_t,
            'epsilon_start_val': epsilon_start_val,
            'epsilon_end_val': epsilon_end_val,
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'current_epsilon': current_epsilon,
            'discount_factor': self.discount_factor,
            'update_target_estimator_every': self.update_target_estimator_every,
            'batch_size': self.batch_size,
            'num_actions': self.num_actions,
            'train_every': self.train_every,
            'device': self.device
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' Restore the model from a checkpoint. '''
        print("\nINFO - Restoring LSTM-VAE DQN model from checkpoint...")
        q_estimator_attrs = checkpoint['q_estimator']

        agent_instance = cls(
            replay_memory_size=checkpoint['memory']['memory_size'],
            update_target_estimator_every=checkpoint['update_target_estimator_every'],
            discount_factor=checkpoint['discount_factor'],
            epsilon_start=checkpoint['epsilon_start_val'],
            epsilon_end=checkpoint['epsilon_end_val'],
            epsilon_decay_steps=checkpoint['epsilon_decay_steps'],
            batch_size=checkpoint['batch_size'],
            num_actions=checkpoint['num_actions'],
            state_shape=q_estimator_attrs['state_shape'],
            train_every=checkpoint['train_every'],
            lstm_input_size=q_estimator_attrs['lstm_input_size'],
            lstm_hidden_size=q_estimator_attrs['lstm_hidden_size'],
            lstm_num_layers=q_estimator_attrs['lstm_num_layers'],
            latent_dim=q_estimator_attrs['latent_dim'],
            kld_weight=q_estimator_attrs['kld_weight'],
            learning_rate=q_estimator_attrs['learning_rate'],
            device=checkpoint['device']
        )

        agent_instance.total_t = checkpoint['total_t']
        agent_instance.train_t = checkpoint['train_t']

        agent_instance.q_estimator = Estimator.from_checkpoint(q_estimator_attrs)

        estimator_args = {
            'num_actions': agent_instance.num_actions,
            'learning_rate': agent_instance.q_estimator.learning_rate,
            'state_shape': agent_instance.q_estimator.state_shape,
            'device': agent_instance.device,
            'lstm_input_size': agent_instance.q_estimator.lstm_input_size,
            'lstm_hidden_size': agent_instance.q_estimator.lstm_hidden_size,
            'lstm_num_layers': agent_instance.q_estimator.lstm_num_layers,
            'latent_dim': agent_instance.q_estimator.latent_dim,
            'kld_weight': agent_instance.q_estimator.kld_weight
        }
        agent_instance.target_estimator = Estimator(**estimator_args)
        agent_instance.target_estimator.qnet.load_state_dict(agent_instance.q_estimator.qnet.state_dict())
        agent_instance.target_estimator.qnet.eval()

        agent_instance.memory = Memory.from_checkpoint(checkpoint['memory'])

        print(f"INFO - Model restored. total_t={agent_instance.total_t}, train_t={agent_instance.train_t}")
        return agent_instance

    def save_checkpoint(self, path, filename='checkpoint_dqn.pt'):
        ''' Save the model checkpoint '''
        # User must ensure the path directory exists.
        filepath = path + '/' + filename # Using simple concatenation
        try:
            torch.save(self.checkpoint_attributes(), filepath)
        except Exception as e:
             print(f"Error saving checkpoint to {filepath}: {e}")


class Estimator(object):
    '''
    Q-Value Estimator using the LSTMEstimatorNetwork.
    Manages network training including Q-loss and KLD loss.
    '''
    def __init__(self, num_actions=2, learning_rate=0.001, state_shape=None,
                 device=None,
                 lstm_input_size=None, lstm_hidden_size=128, lstm_num_layers=1,
                 latent_dim=32, kld_weight=0.001):
        
        self.num_actions = num_actions
        self.learning_rate=learning_rate
        self.state_shape = state_shape
        self.device = device
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight

        # Setup Q model using LSTMEstimatorNetwork
        qnet = LSTMEstimatorNetwork(num_actions, state_shape,
                                    lstm_input_size, lstm_hidden_size, lstm_num_layers,
                                    latent_dim, device)
        qnet = qnet.to(self.device)
        self.qnet = qnet
        self.qnet.eval()

        # Initialize weights
        for name, p in self.qnet.named_parameters():
            if 'weight' in name and 'lstm' not in name and len(p.data.shape) > 1:
                 nn.init.xavier_uniform_(p.data)
            elif 'bias' in name:
                 nn.init.constant_(p.data, 0)

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

    def predict_nograd(self, s):
        ''' Predicts action values deterministically (using mu), without gradients. '''
        with torch.no_grad():
            s_tensor = torch.from_numpy(s).float().to(self.device)
            q_as, _, _ = self.qnet(s_tensor, sample_latent=False)
            q_as_numpy = q_as.cpu().numpy()
        return q_as_numpy

    def update(self, s, a, y):
        ''' Updates the estimator: calculates combined loss (Q + KLD) and backpropagates. '''
        self.optimizer.zero_grad()
        self.qnet.train()

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        q_as_pred, mu, log_var = self.qnet(s, sample_latent=True)

        Q_pred = torch.gather(q_as_pred, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)
        q_loss = self.mse_loss(Q_pred, y)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

        total_loss = q_loss + self.kld_weight * kld_loss
        total_loss.backward()
        self.optimizer.step()

        q_loss_item = q_loss.item()
        self.qnet.eval()
        return q_loss_item

    def checkpoint_attributes(self):
        ''' Return the attributes needed to restore the model from a checkpoint. '''
        return {
            'qnet': self.qnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'state_shape': self.state_shape,
            'lstm_input_size': self.lstm_input_size,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'latent_dim': self.latent_dim,
            'kld_weight': self.kld_weight,
            'device': self.device
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' Restore the model from a checkpoint. '''
        estimator = cls(
            num_actions=checkpoint['num_actions'],
            learning_rate=checkpoint['learning_rate'],
            state_shape=checkpoint['state_shape'],
            device=checkpoint['device'],
            lstm_input_size=checkpoint['lstm_input_size'],
            lstm_hidden_size=checkpoint['lstm_hidden_size'],
            lstm_num_layers=checkpoint['lstm_num_layers'],
            latent_dim=checkpoint['latent_dim'],
            kld_weight=checkpoint['kld_weight']
        )
        estimator.qnet.load_state_dict(checkpoint['qnet'])
        estimator.optimizer.load_state_dict(checkpoint['optimizer'])
        return estimator


class LSTMEstimatorNetwork(nn.Module):
    '''
    An LSTM-based VAE-like network structure for the Q-Value Estimator.
    '''
    def __init__(self, num_actions=2, state_shape=None,
                 lstm_input_size=None, lstm_hidden_size=128, lstm_num_layers=1,
                 latent_dim=32, device=None):
        
        super(LSTMEstimatorNetwork, self).__init__()
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.flat_state_dim = np.prod(self.state_shape)
        self.lstm_input_size = lstm_input_size if lstm_input_size is not None else self.flat_state_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.latent_dim = latent_dim
        self.device = device

        if self.lstm_input_size != self.flat_state_dim:
            self.input_proj = nn.Linear(self.flat_state_dim, self.lstm_input_size)
            self.input_activation = nn.Tanh()
        else:
            self.input_proj = None
            self.input_activation = None

        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers,
                            batch_first=True)

        self.fc_mu = nn.Linear(self.lstm_hidden_size, self.latent_dim)
        self.fc_log_var = nn.Linear(self.lstm_hidden_size, self.latent_dim)
        self.fc_q = nn.Linear(self.latent_dim, self.num_actions)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, s, sample_latent=True):
        ''' Forward pass: state -> [projection] -> LSTM -> mu, log_var -> z -> Q-values '''
        batch_size = s.size(0)
        s_flat = s.view(batch_size, -1)

        if self.input_proj:
            s_processed = self.input_proj(s_flat)
            if self.input_activation:
                 s_processed = self.input_activation(s_processed)
        else:
            s_processed = s_flat

        s_lstm_input = s_processed.unsqueeze(1)

        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(s_lstm_input.device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(s_lstm_input.device)
        lstm_out, _ = self.lstm(s_lstm_input, (h0, c0))
        lstm_out_last = lstm_out[:, -1, :]

        mu = self.fc_mu(lstm_out_last)
        log_var = self.fc_log_var(lstm_out_last)

        if sample_latent:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu

        q_values = self.fc_q(z)
        return q_values, mu, log_var


class Memory(object):
    ''' Memory for saving transitions '''
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, legal_actions, done):
        ''' Save transition into memory '''
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done, legal_actions)
        self.memory.append(transition)

    def sample(self):
        ''' Sample a minibatch from the replay memory '''
        samples = random.sample(self.memory, self.batch_size)
        try:
            samples_unzipped = tuple(zip(*samples))
            return tuple(map(np.array, samples_unzipped[:-1])) + (samples_unzipped[-1],)
        except ValueError as e:
            print(f"Error converting samples to numpy arrays: {e}")
            print("Check if all 'state' and 'next_state' observations in the memory have the same shape.")
            raise e
        except Exception as e:
            print(f"Error during memory sampling: {e}")
            raise e

    def checkpoint_attributes(self):
        ''' Returns the attributes that need to be checkpointed '''
        return {
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'memory': self.memory
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' Restores the attributes from the checkpoint '''
        instance = cls(checkpoint['memory_size'], checkpoint['batch_size'])
        instance.memory = checkpoint['memory']
        return instance


# Example Usage (Illustrative - No external libraries like os used here)
if __name__ == '__main__':
    state_dim = 56
    num_actions = 5

    agent = VAEDQNAgent(
        state_shape=[state_dim],
        num_actions=num_actions,
        learning_rate=0.0005,
        lstm_hidden_size=64,
        lstm_num_layers=1,
        latent_dim=20,
        kld_weight=0.005,
        batch_size=64,
        replay_memory_init_size=100,
        update_target_estimator_every=500,
        save_path='./saved_models_lstm_vae', # Example path
        save_every=1000
    )

    print("Agent initialized with LSTM-VAE network.")
    print(f"Using device: {agent.device}")

    print("\nStarting dummy training loop...")
    step_count = 0
    train_step_count = 0
    for episode in range(5):
        legal_act_keys = random.sample(range(num_actions), k=random.randint(1, num_actions))
        current_state = {
            'obs': np.random.rand(state_dim).astype(np.float32),
            'legal_actions': {act: None for act in legal_act_keys},
            'raw_legal_actions': legal_act_keys
        }
        done = False
        ep_steps = 0
        while not done and ep_steps < 50:
            action = agent.step(current_state)
            reward = random.uniform(-1, 1)
            ep_steps += 1
            step_count += 1
            done = (ep_steps == 50)

            if not done:
                next_legal_act_keys = random.sample(range(num_actions), k=random.randint(1, num_actions))
                next_state = {
                    'obs': np.random.rand(state_dim).astype(np.float32),
                    'legal_actions': {act: None for act in next_legal_act_keys},
                    'raw_legal_actions': next_legal_act_keys
                }
            else:
                 next_state = {
                    'obs': np.zeros(state_dim).astype(np.float32),
                    'legal_actions': {},
                    'raw_legal_actions': []
                 }

            transition = (current_state, action, reward, next_state, done)
            agent.feed(transition)
            current_state = next_state

            if agent.total_t > agent.replay_memory_init_size and \
               (agent.total_t - agent.replay_memory_init_size) % agent.train_every == 0:
                train_step_count +=1

        print(f"Episode {episode+1} finished. Total steps: {step_count}. Approx train steps: {train_step_count}")

    print("\nDummy training loop finished.")

    print("\nSaving checkpoint...")
    # Important: Ensure the directory agent.save_path exists before calling this
    agent.save_checkpoint(agent.save_path, filename='final_lstm_vae_checkpoint.pt')

    print("Loading checkpoint...")
    checkpoint_filepath = agent.save_path + '/' + 'final_lstm_vae_checkpoint.pt' # Simple concatenation
    # Important: Check if file exists before trying to load
    try:
        # Load requires file to exist
        checkpoint_data = torch.load(checkpoint_filepath, map_location=agent.device)
        restored_agent = VAEDQNAgent.from_checkpoint(checkpoint_data)
        print("Agent restored successfully.")
        print(f"Restored agent train_t: {restored_agent.train_t}")
    except FileNotFoundError:
         print(f"Checkpoint file not found at {checkpoint_filepath}. Make sure the path exists and saving was successful.")
    except Exception as e:
         print(f"Error loading checkpoint from {checkpoint_filepath}: {e}")