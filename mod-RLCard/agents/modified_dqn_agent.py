import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy

# Keep rlcard import if you are using it, otherwise remove if standalone
# from rlcard.utils.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ New LSTM VAE Estimator Network +++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class LSTMEstimatorNetwork(nn.Module):
    '''
    An LSTM-based VAE-like network structure for the Q-Value Estimator.
    It processes the state with an LSTM, uses the LSTM's output to define
    a latent distribution (mu, log_var), samples from it (z),
    and then predicts Q-values from the latent sample z.
    '''
    def __init__(self, num_actions=2, state_shape=None,
                 lstm_input_size=None, lstm_hidden_size=128, lstm_num_layers=1,
                 latent_dim=32, device=None):
        ''' Initialize the LSTM-VAE Q network

        Args:
            num_actions (int): number of legal actions
            state_shape (list): shape of state tensor
            lstm_input_size (int, optional): Dimension state is projected to before LSTM.
                                            If None, uses flattened state size.
            lstm_hidden_size (int): Number of features in the hidden state of the LSTM.
            lstm_num_layers (int): Number of recurrent layers in the LSTM.
            latent_dim (int): Dimension of the VAE latent space.
            device (torch.device): CPU or GPU device.
        '''
        super(LSTMEstimatorNetwork, self).__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.flat_state_dim = np.prod(self.state_shape)
        self.lstm_input_size = lstm_input_size if lstm_input_size is not None else self.flat_state_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.latent_dim = latent_dim
        self.device = device # Store device

        # --- Input Processing Layer (Optional) ---
        # If lstm_input_size is specified and different from flat_state_dim,
        # add a linear layer to project the state.
        if self.lstm_input_size != self.flat_state_dim:
            self.input_proj = nn.Linear(self.flat_state_dim, self.lstm_input_size)
            # Activation after projection? Tanh is used elsewhere.
            self.input_activation = nn.Tanh()
        else:
            self.input_proj = None
            self.input_activation = None

        # --- LSTM Layer ---
        # batch_first=True expects input: (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers,
                            batch_first=True)

        # --- Latent Space Layers ---
        # Takes LSTM hidden state output to mu and log_var
        self.fc_mu = nn.Linear(self.lstm_hidden_size, self.latent_dim)
        self.fc_log_var = nn.Linear(self.lstm_hidden_size, self.latent_dim)

        # --- Q-Value Prediction Layer ---
        # Takes sampled latent variable z to Q-values
        self.fc_q = nn.Linear(self.latent_dim, self.num_actions)

    def reparameterize(self, mu, log_var):
        ''' Performs reparameterization trick '''
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # Sample epsilon from N(0, I)
        return mu + eps * std

    def forward(self, s, sample_latent=True):
        ''' Forward pass: state -> [projection] -> LSTM -> mu, log_var -> z -> Q-values

        Args:
            s (Tensor): Input state tensor (batch, state_shape)
            sample_latent (bool): If True, sample z using reparameterization trick.
                                  If False, use mu as z (deterministic for prediction).

        Returns:
            q_values (Tensor): (batch, num_actions)
            mu (Tensor): (batch, latent_dim)
            log_var (Tensor): (batch, latent_dim)
        '''
        # 1. Flatten and potentially project input state
        batch_size = s.size(0)
        s_flat = s.view(batch_size, -1) # Flatten state

        if self.input_proj:
            s_processed = self.input_proj(s_flat)
            if self.input_activation:
                 s_processed = self.input_activation(s_processed)
        else:
            s_processed = s_flat

        # 2. Reshape for LSTM: (batch, seq_len, features)
        # We treat the single state observation as a sequence of length 1
        s_lstm_input = s_processed.unsqueeze(1) # Shape: (batch, 1, lstm_input_size)

        # 3. Pass through LSTM
        # Initialize hidden and cell states
        # Shape: (num_layers, batch, hidden_size)
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(self.device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(self.device)

        # LSTM forward pass
        # lstm_out shape: (batch, seq_len, hidden_size) -> (batch, 1, hidden_size)
        # h_n shape: (num_layers, batch, hidden_size) -> final hidden state
        # c_n shape: (num_layers, batch, hidden_size) -> final cell state
        lstm_out, (h_n, c_n) = self.lstm(s_lstm_input, (h0, c0))

        # 4. Get LSTM output for VAE part
        # Use the output of the last time step (which is the only time step here)
        lstm_out_last = lstm_out[:, -1, :] # Shape: (batch, hidden_size)
        # Alternatively, could use the final hidden state: h_n[-1] which should be the same for seq_len=1

        # 5. Calculate mu and log_var
        mu = self.fc_mu(lstm_out_last)
        log_var = self.fc_log_var(lstm_out_last)

        # 6. Sample from latent distribution z (or use mu deterministically)
        if sample_latent:
            z = self.reparameterize(mu, log_var) # Shape: (batch, latent_dim)
        else:
            z = mu # Use mean for deterministic prediction

        # 7. Predict Q-values from latent variable z
        q_values = self.fc_q(z) # Shape: (batch, num_actions)

        return q_values, mu, log_var


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Updated Estimator Class ++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Estimator(object):
    '''
    Q-Value Estimator using the LSTMEstimatorNetwork.
    '''
    # *** MODIFIED __init__ for LSTM-VAE parameters ***
    def __init__(self, num_actions=2, learning_rate=0.001, state_shape=None,
                 lstm_input_size=None, lstm_hidden_size=128, lstm_num_layers=1,
                 latent_dim=32, device=None): # Replaced encoder/decoder with LSTM/latent params
        ''' Initilalize an Estimator object.

        Args:
            num_actions (int): the number output actions
            learning_rate (float): The learning rate
            state_shape (list): the shape of the state space
            lstm_input_size (int, optional): Dimension state is projected to before LSTM.
            lstm_hidden_size (int): LSTM hidden size.
            lstm_num_layers (int): Number of LSTM layers.
            latent_dim (int): Dimension of the VAE latent space.
            device (torch.device): whether to use cpu or gpu
        '''
        self.num_actions = num_actions
        self.learning_rate=learning_rate
        self.state_shape = state_shape
        # Store LSTM VAE architecture params
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.latent_dim = latent_dim
        self.device = device

        # Set up Q model using the new LSTM VAE network and place it in eval mode
        # *** USE LSTMEstimatorNetwork ***
        qnet = LSTMEstimatorNetwork(num_actions, state_shape,
                                    lstm_input_size, lstm_hidden_size, lstm_num_layers,
                                    latent_dim, device) # Pass device to network
        qnet = qnet.to(self.device)
        self.qnet = qnet
        self.qnet.eval() # Set to evaluation mode initially

        # Initialize the weights using Xavier init (might need adjustment for LSTM parts)
        for name, p in self.qnet.named_parameters():
            # Xavier init for Linear layers' weights
            if 'weight' in name and 'lstm' not in name and len(p.data.shape) > 1:
                 nn.init.xavier_uniform_(p.data)
            # Standard init for LSTM weights (PyTorch default is usually okay)
            elif 'lstm' in name and 'weight' in name:
                 # Optionally apply orthogonal or xavier init to LSTM weights if needed
                 # nn.init.orthogonal_(p.data)
                 pass # Keep default LSTM init for now
            # Zero init for biases
            elif 'bias' in name:
                 nn.init.constant_(p.data, 0)

        # Set up Q-loss function (MSE for Q-learning part)
        self.mse_loss = nn.MSELoss(reduction='mean')

        # Set up optimizer
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

        # --- Placeholder for potential VAE Loss (KL Divergence) ---
        # Define KLD loss function if needed later
        self.kld_weight = 0.001 # Example weight
        def kld_loss_fn(mu, log_var):
            # KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            # Average over batch dimension
            return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        self.kld_loss_fn = kld_loss_fn
        # ---------------------------------------------------------


    # *** MODIFIED predict_nograd to use sample_latent=False ***
    def predict_nograd(self, s):
        ''' Predicts action values deterministically using mu, not sampling z.
            Prediction is not included in the computation graph.
        '''
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            # *** Use sample_latent=False for deterministic prediction ***
            q_as, _, _ = self.qnet(s, sample_latent=False)
            q_as_numpy = q_as.cpu().numpy()
        return q_as_numpy

    # *** MODIFIED update to handle network output and potentially add KLD loss ***
    def update(self, s, a, y):
        ''' Updates the estimator towards the given targets (y).
            Includes sampling from the latent space (sample_latent=True).
            Optionally includes KL divergence loss.
        '''
        self.optimizer.zero_grad()
        self.qnet.train() # Set to training mode

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # Forward pass - use sample_latent=True for training
        # Returns q_values, mu, log_var
        q_as_pred, mu, log_var = self.qnet(s, sample_latent=True)

        # --- Calculate Q-Learning Loss (MSE) ---
        Q_pred = torch.gather(q_as_pred, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)
        q_loss = self.mse_loss(Q_pred, y)

        # --- (Optional) Calculate VAE KL Divergence Loss ---
        kld_loss = self.kld_loss_fn(mu, log_var)
        # You might want separate learning rates or just different weights
        total_loss = q_loss + self.kld_weight * kld_loss # Combine losses
        # ----------------------------------------------------

        # --- Backpropagate and Update ---
        # Use total_loss if KLD is included, otherwise just q_loss
        batch_loss = total_loss # or batch_loss = q_loss if KLD not used
        batch_loss.backward()

        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=1.0)

        self.optimizer.step()
        q_loss_item = q_loss.item()
        kld_loss_item = kld_loss.item() # Get KLD loss value if used

        self.qnet.eval() # Set back to evaluation mode

        # Return Q-loss and KLD loss separately for logging if desired
        return q_loss_item, kld_loss_item


    # *** MODIFIED checkpoint_attributes for LSTM-VAE params ***
    def checkpoint_attributes(self):
        ''' Return the attributes needed to restore the model from a checkpoint
        '''
        return {
            'qnet': self.qnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'state_shape': self.state_shape,
            # Save LSTM VAE architecture params
            'lstm_input_size': self.lstm_input_size,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'latent_dim': self.latent_dim,
            'device': self.device,
            'kld_weight': self.kld_weight # Save KLD weight if used
        }

    # *** MODIFIED from_checkpoint for LSTM-VAE params ***
    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' Restore the model from a checkpoint
        '''
        estimator = cls(
            num_actions=checkpoint['num_actions'],
            learning_rate=checkpoint['learning_rate'],
            state_shape=checkpoint['state_shape'],
            # Restore LSTM VAE architecture params
            lstm_input_size=checkpoint['lstm_input_size'],
            lstm_hidden_size=checkpoint['lstm_hidden_size'],
            lstm_num_layers=checkpoint['lstm_num_layers'],
            latent_dim=checkpoint['latent_dim'],
            device=checkpoint['device']
        )
        # Restore KLD weight if saved
        if 'kld_weight' in checkpoint:
            estimator.kld_weight = checkpoint['kld_weight']

        estimator.qnet.load_state_dict(checkpoint['qnet'])
        estimator.optimizer.load_state_dict(checkpoint['optimizer'])
        return estimator


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Updated DQNAgent Class +++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class VAEDQNAgent(object):
    '''
    DQN Agent using an LSTM-VAE Estimator network.
    '''
    # *** MODIFIED __init__ for LSTM-VAE parameters ***
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
                 # Replace VAE params with LSTM-VAE params
                 lstm_input_size=None, # Optional projection size
                 lstm_hidden_size=128, # LSTM hidden dim
                 lstm_num_layers=1,    # LSTM layers
                 latent_dim=32,        # Latent dim for VAE part
                 learning_rate=0.00005,
                 device=None,
                 save_path=None,
                 save_every=float('inf'),
                 kld_weight=0.001 # Add KLD weight param here if you want to control it from agent init
                 ):

        ''' Q-Learning algorithm... (docstring updated for LSTM-VAE)

        Args:
            ... (other args) ...
            lstm_input_size (int, optional): The dimension to project state features to before LSTM.
            lstm_hidden_size (int): The hidden dimension size for the LSTM.
            lstm_num_layers (int): The number of layers for the LSTM.
            latent_dim (int): The dimension of the VAE latent space.
            kld_weight (float): Weight for the KL divergence loss term.
            ... (other args) ...
        '''
        self.use_raw = False # Assuming this is from rlcard context
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

        # *** Create estimators using LSTM-VAE parameters ***
        self.q_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape,
                                     lstm_input_size=lstm_input_size, lstm_hidden_size=lstm_hidden_size,
                                     lstm_num_layers=lstm_num_layers, latent_dim=latent_dim,
                                     device=self.device)
        # Set KLD weight in the estimator
        self.q_estimator.kld_weight = kld_weight

        # Target estimator should have the same architecture
        self.target_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape,
                                          lstm_input_size=lstm_input_size, lstm_hidden_size=lstm_hidden_size,
                                          lstm_num_layers=lstm_num_layers, latent_dim=latent_dim,
                                          device=self.device)
        self.target_estimator.kld_weight = kld_weight # Keep consistent if needed, though target doesn't use KLD loss

        # Ensure target estimator starts as a deep copy of q_estimator's initial state
        # self.target_estimator = deepcopy(self.q_estimator) # Incorrect deepcopy here, copy state_dict after init
        self.target_estimator.qnet.load_state_dict(self.q_estimator.qnet.state_dict())


        self.memory = Memory(replay_memory_size, batch_size)
        self.save_path = save_path
        self.save_every = save_every

    # --- Methods feed, step, eval_step remain the same ---
    # --- They interact via predict which uses the estimator ---
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
        q_values = self.predict(state) # Uses predict_nograd (deterministic mu)
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        legal_actions = list(state['legal_actions'].keys())

        if not legal_actions: # Handle cases with no legal actions
             # This case should ideally not happen in standard environments after a non-terminal state
             print("Warning: No legal actions available in step(). Choosing action 0.")
             return 0 # Or raise an error, or choose a default 'pass' action if available

        if random.random() < epsilon:
            action = random.choice(legal_actions)
        else:
            # q_values is already masked, find max among legal actions
            # Ensure q_values for legal actions are finite before argmax
            q_subset = q_values[legal_actions]
            if np.all(np.isinf(q_subset)):
                 # If all legal actions have -inf Q-value (shouldn't happen if predict works)
                 # Choose randomly among legal actions as a fallback
                 # print("Warning: All legal actions have -inf Q-value. Choosing randomly.")
                 action = random.choice(legal_actions)
            else:
                 action = legal_actions[np.argmax(q_subset)]

        return action


    def eval_step(self, state):
        ''' Predict the best action for evaluation (greedy). '''
        q_values = self.predict(state) # Uses predict_nograd (deterministic mu)
        legal_actions = list(state['legal_actions'].keys())

        if not legal_actions:
             print("Warning: No legal actions available in eval_step(). Choosing action 0.")
             best_action = 0 # Or handle as error/default action
        else:
             q_subset = q_values[legal_actions]
             if np.all(np.isinf(q_subset)):
                  # print("Warning: All legal actions have -inf Q-value in eval_step. Choosing action 0.")
                  best_action = 0 # Or random choice fallback
             else:
                  best_action = legal_actions[np.argmax(q_subset)]

        info = {}
        raw_legal_actions = state.get('raw_legal_actions', legal_actions)
        # Ensure info values are finite, replace -inf with a very small number if needed
        legal_q_vals_float = {raw_legal_actions[i]: float(q_values[legal_actions[i]]) if np.isfinite(q_values[legal_actions[i]]) else -1e9 for i in range(len(legal_actions))}
        info['values'] = legal_q_vals_float

        return best_action, info

    # --- Method predict remains the same ---
    # --- It uses estimator.predict_nograd which is updated ---
    def predict(self, state):
        ''' Predict the masked Q-values using deterministic network output (mu). '''
        obs = np.expand_dims(state['obs'], 0)
        # predict_nograd now uses sample_latent=False internally
        q_values = self.q_estimator.predict_nograd(obs)[0] # Shape (num_actions,)

        legal_actions = list(state['legal_actions'].keys())
        masked_q_values = np.full(self.num_actions, -np.inf, dtype=float) # Mask with -inf

        # Avoid error if legal_actions is empty or contains invalid indices
        valid_legal_actions = [a for a in legal_actions if 0 <= a < self.num_actions]
        if not valid_legal_actions:
             # Handle case with no valid legal actions if necessary
             # print("Warning: No valid legal actions found in predict.")
             pass # masked_q_values remains all -inf
        else:
             masked_q_values[valid_legal_actions] = q_values[valid_legal_actions]


        return masked_q_values


    # --- Train method updated to handle KLD loss return value ---
    def train(self):
        ''' Train the network, potentially including KLD loss. '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()

        # 1. Calculate best next actions using the *online* Q-network (Double DQN)
        # Uses predict_nograd -> sample_latent=False (deterministic mu)
        q_values_next_online = self.q_estimator.predict_nograd(next_state_batch)

        best_actions = np.zeros(self.batch_size, dtype=int)
        for i in range(self.batch_size):
            legal_actions = legal_actions_batch[i]
            if not legal_actions:
                 best_actions[i] = 0 # Assign default action for terminal states
            else:
                 valid_legal = [a for a in legal_actions if 0 <= a < self.num_actions]
                 if not valid_legal:
                      best_actions[i] = 0 # Default if no valid actions
                 else:
                      masked_q = np.full(self.num_actions, -np.inf)
                      masked_q[valid_legal] = q_values_next_online[i, valid_legal]
                      best_actions[i] = np.argmax(masked_q)


        # 2. Evaluate Q-value of these best actions using the *target* network
        # Uses predict_nograd -> sample_latent=False (deterministic mu)
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        q_target_values = q_values_next_target[np.arange(self.batch_size), best_actions]

        # 3. Calculate the TD target
        td_target = reward_batch + (np.invert(done_batch).astype(np.float32) *
                                   self.discount_factor * q_target_values)

        # 4. Perform gradient descent update on the online Q-network
        # Includes sampling (sample_latent=True) and KLD loss calculation inside estimator.update
        state_batch_np = np.array(state_batch)
        # *** Estimator update now returns q_loss and kld_loss ***
        q_loss, kld_loss = self.q_estimator.update(state_batch_np, action_batch, td_target)

        # Print both losses for monitoring
        print(f'\rINFO - Step {self.total_t}, Q-Loss: {q_loss:.4f}, KLD-Loss: {kld_loss:.4f}', end='')

        # 5. Update the target estimator periodically
        if self.train_t > 0 and self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator.qnet.load_state_dict(self.q_estimator.qnet.state_dict())
            print(f"\nINFO - Copied model parameters to target network at train step {self.train_t}.")

        self.train_t += 1

        # 6. Save model checkpoint periodically
        if self.save_path and self.train_t > 0 and self.train_t % self.save_every == 0:
            self.save_checkpoint(self.save_path)
            print(f"\nINFO - Saved model checkpoint at train step {self.train_t}.")

        # Return the primary RL loss (Q-loss)
        return q_loss


    # --- feed_memory, set_device remain the same ---
    def feed_memory(self, state, action, reward, next_state, legal_actions, done):
        ''' Feed transition to memory '''
        self.memory.save(state, action, reward, next_state, legal_actions, done)

    def set_device(self, device):
        self.device = device
        self.q_estimator.device = device
        self.q_estimator.qnet.to(device) # Ensure network is on the right device
        self.target_estimator.device = device
        self.target_estimator.qnet.to(device) # Ensure target network is on the right device


    # *** MODIFIED checkpoint_attributes for LSTM-VAE params ***
    def checkpoint_attributes(self):
        ''' Return the current checkpoint attributes (dict) '''
        q_estimator_attrs = self.q_estimator.checkpoint_attributes()
        # Target estimator state can be inferred by copying q_estimator, but saving explicitly is safer if they diverge
        # target_estimator_attrs = self.target_estimator.checkpoint_attributes()

        return {
            'agent_type': 'DQNAgentLSTMVAE', # More specific type
            'q_estimator': q_estimator_attrs, # Contains LSTM VAE params now
            # 'target_estimator': target_estimator_attrs, # Optionally save target state too
            'memory': self.memory.checkpoint_attributes(),
            'total_t': self.total_t,
            'train_t': self.train_t,
            'current_epsilon': self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)],
            'epsilon_start_val': self.epsilons[0],
            'epsilon_end': self.epsilon_end, # Assuming epsilon_end is stored/accessible
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'discount_factor': self.discount_factor,
            'update_target_estimator_every': self.update_target_estimator_every,
            'batch_size': self.batch_size,
            'num_actions': self.num_actions,
            'train_every': self.train_every,
            'device': self.device,
            # Save agent-level KLD weight if it's controlled here
            'kld_weight': self.q_estimator.kld_weight
        }

    # *** MODIFIED from_checkpoint for LSTM-VAE params ***
    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' Restore the model from a checkpoint '''
        print("\nINFO - Restoring LSTM-VAE DQN model from checkpoint...")

        agent_instance = cls(
            replay_memory_size=checkpoint['memory']['memory_size'],
            update_target_estimator_every=checkpoint['update_target_estimator_every'],
            discount_factor=checkpoint['discount_factor'],
            epsilon_start=checkpoint.get('epsilon_start_val', checkpoint.get('current_epsilon', 1.0)),
            epsilon_end=checkpoint['epsilon_end'],
            epsilon_decay_steps=checkpoint['epsilon_decay_steps'],
            batch_size=checkpoint['batch_size'],
            num_actions=checkpoint['num_actions'],
            device=checkpoint['device'],
            state_shape=checkpoint['q_estimator']['state_shape'],
            # Pass LSTM VAE parameters from the saved q_estimator attributes
            lstm_input_size=checkpoint['q_estimator']['lstm_input_size'],
            lstm_hidden_size=checkpoint['q_estimator']['lstm_hidden_size'],
            lstm_num_layers=checkpoint['q_estimator']['lstm_num_layers'],
            latent_dim=checkpoint['q_estimator']['latent_dim'],
            learning_rate=checkpoint['q_estimator']['learning_rate'],
            train_every=checkpoint['train_every'],
            # Restore KLD weight
            kld_weight=checkpoint.get('kld_weight', 0.001) # Default if not found
        )

        agent_instance.total_t = checkpoint['total_t']
        agent_instance.train_t = checkpoint['train_t']
        agent_instance.epsilons = np.linspace(agent_instance.epsilons[0], agent_instance.epsilon_end, agent_instance.epsilon_decay_steps)

        # Restore estimators
        agent_instance.q_estimator = Estimator.from_checkpoint(checkpoint['q_estimator'])
        # Restore target estimator - create a new one and load state dict
        # This ensures it has the correct architecture before loading weights
        agent_instance.target_estimator = Estimator(
             num_actions=agent_instance.num_actions, learning_rate=agent_instance.q_estimator.learning_rate,
             state_shape=agent_instance.q_estimator.state_shape,
             lstm_input_size=agent_instance.q_estimator.lstm_input_size,
             lstm_hidden_size=agent_instance.q_estimator.lstm_hidden_size,
             lstm_num_layers=agent_instance.q_estimator.lstm_num_layers,
             latent_dim=agent_instance.q_estimator.latent_dim,
             device=agent_instance.device
        )
        agent_instance.target_estimator.kld_weight = agent_instance.q_estimator.kld_weight
        # Load the state dict from the Q estimator at the time of saving
        # If target_estimator state was saved separately, load that instead.
        agent_instance.target_estimator.qnet.load_state_dict(agent_instance.q_estimator.qnet.state_dict())


        agent_instance.memory = Memory.from_checkpoint(checkpoint['memory'])

        print(f"INFO - Model restored. total_t={agent_instance.total_t}, train_t={agent_instance.train_t}")
        return agent_instance


    # --- save_checkpoint remains the same, maybe update filename ---
    def save_checkpoint(self, path, filename='checkpoint_dqn_lstm_vae.pt'): # Changed default filename
        ''' Save the model checkpoint (all attributes) '''
        import os
        os.makedirs(path, exist_ok=True)
        try:
            torch.save(self.checkpoint_attributes(), os.path.join(path, filename))
        except Exception as e:
             print(f"Error saving checkpoint: {e}")
             # Consider saving components separately if the full dict fails
             # torch.save(self.q_estimator.qnet.state_dict(), os.path.join(path, 'qnet_state.pt'))
             # ... save other components


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Memory Class (Remains Unchanged) +++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Memory(object):
    ''' Memory for saving transitions '''
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, legal_actions, done):
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done, legal_actions)
        self.memory.append(transition)

    def sample(self):
        samples = random.sample(self.memory, self.batch_size)
        try:
            samples_unzipped = tuple(zip(*samples))
            return tuple(map(np.array, samples_unzipped[:-1])) + (samples_unzipped[-1],)
        except Exception as e:
            print(f"Error during memory sampling - check data consistency: {e}")
            # Detailed inspection if needed
            # for i, s in enumerate(samples):
            #     print(f"Sample {i}: state type {type(s.state)}, action type {type(s.action)}, reward type {type(s.reward)}, next_state type {type(s.next_state)}, done type {type(s.done)}, legal_actions type {type(s.legal_actions)}")
            #     if isinstance(s.state, np.ndarray): print(f"  State shape: {s.state.shape}")
            #     if isinstance(s.next_state, np.ndarray): print(f"  Next State shape: {s.next_state.shape}")
            raise e


    def checkpoint_attributes(self):
        return {
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'memory': self.memory
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        instance = cls(checkpoint['memory_size'], checkpoint['batch_size'])
        instance.memory = checkpoint['memory']
        return instance

# Example usage modification (if you run this standalone)
if __name__ == '__main__':
    # Example parameters
    state_dim = 56 # Example state dimension
    num_actions = 5 # Example number of actions

    agent = VAEDQNAgent(
        state_shape=[state_dim], # Must be a list/tuple
        num_actions=num_actions,
        learning_rate=0.0005,
        lstm_hidden_size=64,   # Specify LSTM hidden size
        lstm_num_layers=1,     # Specify LSTM layers
        latent_dim=20,         # Specify latent dimension
        kld_weight=0.005,      # Specify KLD weight
        batch_size=64,
        replay_memory_init_size=100,
        update_target_estimator_every=500,
        save_path='./saved_models',
        save_every=1000
    )

    print("Agent initialized with LSTM-VAE network.")
    print("Q-Estimator Network:", agent.q_estimator.qnet)
    print(f"Using device: {agent.device}")

    # --- Dummy Training Loop Example ---
    print("\nStarting dummy training loop...")
    for i in range(2000): # Simulate more steps
        # Create dummy state dictionary matching rlcard format
        legal_act_keys = random.sample(range(num_actions), k=random.randint(1, num_actions))
        dummy_state = {
            'obs': np.random.rand(state_dim).astype(np.float32),
            'legal_actions': {act: None for act in legal_act_keys},
            'raw_legal_actions': legal_act_keys # Assuming raw corresponds directly
        }

        action = agent.step(dummy_state)
        reward = random.uniform(-1, 1)
        done = ((i + 1) % 50 == 0) # Episode ends every 50 steps

        # Create dummy next state dictionary
        next_legal_act_keys = random.sample(range(num_actions), k=random.randint(1, num_actions))
        dummy_next_state = {
            'obs': np.random.rand(state_dim).astype(np.float32),
            'legal_actions': {act: None for act in next_legal_act_keys},
             'raw_legal_actions': next_legal_act_keys
        } if not done else { # Terminal state might have empty legal actions
             'obs': np.zeros(state_dim).astype(np.float32),
             'legal_actions': {},
             'raw_legal_actions': []
        }

        transition = (dummy_state, action, reward, dummy_next_state, done)
        agent.feed(transition) # feed handles training calls

        if done:
             # Resetting state for next episode not needed here as loop creates new state
             print(f"\nEpisode finished at step {i+1}")
             pass

    print("\nDummy training loop finished.")

     # --- Example saving and loading ---
    print("\nSaving checkpoint...")
    agent.save_checkpoint(agent.save_path)
    print("Loading checkpoint...")
    try:
         # Create path string correctly
         checkpoint_path = os.path.join(agent.save_path, 'checkpoint_dqn_lstm_vae.pt')
         # Load the checkpoint data first
         checkpoint_data = torch.load(checkpoint_path, map_location=agent.device) # Load to the correct device
         # Restore agent from data
         restored_agent = DQNAgent.from_checkpoint(checkpoint_data)
         print("Agent restored successfully.")
         print(f"Restored agent train_t: {restored_agent.train_t}")
    except FileNotFoundError:
         print(f"Checkpoint file not found at {checkpoint_path}")
    except Exception as e:
         print(f"Error loading checkpoint: {e}")