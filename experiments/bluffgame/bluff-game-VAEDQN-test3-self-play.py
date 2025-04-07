''' Training DQN and VAEDQN agent on custom bluff-game environment
'''

import os
import argparse
import random # Needed for opponent selection
import numpy as np # Needed for opponent selection

import torch

import rlcard
# Import necessary agents
from rlcard.agents import RandomAgent
# Import your custom VAE agent - REPLACE 'vaedqn_agent_lstm' with your actual file name
# Make sure this file (e.g., vaedqn_agent_lstm.py) is in the correct path to be imported
from rlcard.agents import VAEDQNAgent as VAEDQNAgent
# Import other standard agents
from rlcard.agents import DQNAgent as StandardDQNAgent
from rlcard.agents import NFSPAgent

from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    MultiLogger,
    plot_curve,
)

# Original train function (kept for reference/potential use)
def train(args):
    # ... (original train function remains exactly as provided before) ...
    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    # Initialize the agent and use ensemble of agents as opponents
    if args.algorithm == 'vaedqn':
        # NOTE: Ensure necessary parameters for VAEDQNAgent are passed or set default
        agent = VAEDQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            device=device,
            # Add other necessary VAEDQN params here, e.g.:
            lstm_hidden_size=64, latent_dim=20, kld_weight=0.005, learning_rate=0.0005
        )
    elif args.algorithm == 'nfsp':
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=[64,64],
            q_mlp_layers=[64,64],
            device=device,
        )
    elif args.algorithm == 'dqn': # Standard DQN
         agent = StandardDQNAgent(
             num_actions=env.num_actions,
             state_shape=env.state_shape[0],
             mlp_layers=[64, 64], # Example MLP layers
             device=device,
         )
    else:
         raise ValueError("Unsupported algorithm specified")

    # Setup default opponents (e.g., RandomAgent) for single train
    opponents = [RandomAgent(num_actions=env.num_actions) for _ in range(1, env.num_players)]
    agents = [agent] + opponents
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env, # evaluates against opponents currently in env
                        args.num_eval_games,
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save model - Use original saving logic
    save_path = os.path.join(args.log_dir, 'model.pth') # Use os.path.join if os is imported
    # save_path = args.log_dir + '/model.pth' # Original style if os not imported
    try:
        torch.save(agent, save_path)
        print('Model saved in', save_path)
    except Exception as e:
        print(f"Error saving model object directly: {e}")
        print("If using a custom agent with checkpointing, consider using its save method.")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Modified Multi-Train Function (Alternating, MultiLogger, Mutual Eval) +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def multi_train(args):
    ''' Train DQN and VAEDQN agents, alternating training.
        Evaluates BOTH agents against RandomAgent AND against each other.
        Uses MultiLogger to log performance vs RandomAgent separately.
        Prints mutual evaluation results to console.
    '''

    device = get_device()
    set_seed(args.seed)
    switch_frequency = 100

    # Get env parameters
    dummy_env = rlcard.make(args.env, config={'seed': args.seed})
    num_actions = dummy_env.num_actions
    state_shape = dummy_env.state_shape[0]
    num_players = dummy_env.num_players
    del dummy_env

    # --- Initialize ALL potential agents ---
    print("Initializing agents...")
    agent_dqn = StandardDQNAgent(
        num_actions=num_actions, state_shape=state_shape,
        mlp_layers=[64, 64], device=device, learning_rate=0.0001,
    )
    agent_vaedqn = VAEDQNAgent(
        num_actions=num_actions, state_shape=state_shape,
        lstm_hidden_size=64, latent_dim=20, kld_weight=0.005,
        learning_rate=0.0005, device=device,
    )
    agent_random = RandomAgent(num_actions=num_actions)
    print("Agents initialized.")

    learning_agents = [agent_dqn, agent_vaedqn]
    agent_keys = ["DQN", "VAE_DQN"] # Keys for MultiLogger

    # Training Loop Setup
    total_episodes_trained = 0
    current_learner_idx = 0

    # Use the new MultiLogger
    with MultiLogger(args.log_dir, agent_keys) as multi_logger:
        while total_episodes_trained < args.num_episodes:

            # Determine current learner and opponent pool
            current_agent = learning_agents[current_learner_idx]
            current_agent_key = agent_keys[current_learner_idx]
            other_learner_agent = learning_agents[1 - current_learner_idx]
            opponent_pool_options = [other_learner_agent, agent_random]
            opponent_agent = random.choice(opponent_pool_options)
            opponent_name = opponent_agent.__class__.__name__
            if opponent_agent == other_learner_agent: opponent_name = agent_keys[1 - current_learner_idx]

            print(f"\n--- Episodes {total_episodes_trained+1} -> {min(total_episodes_trained + switch_frequency, args.num_episodes)} ---")
            print(f"--- Training: {current_agent_key} --- Opponent: {opponent_name} ---")

            # Configure Environment for Training
            env = rlcard.make(args.env, config={'seed': args.seed + total_episodes_trained})
            agents = [current_agent] + [opponent_agent] * (num_players - 1)
            env.set_agents(agents)

            # Run Episodes for this Chunk
            episodes_in_chunk = 0
            while episodes_in_chunk < switch_frequency and total_episodes_trained < args.num_episodes:
                current_episode_num = total_episodes_trained

                # Run environment episode
                trajectories, payoffs = env.run(is_training=True)
                trajectories = reorganize(trajectories, payoffs)

                # Feed transitions only to the current learner
                for ts in trajectories[0]:
                    current_agent.feed(ts)

                # --- Evaluation Step ---
                if total_episodes_trained > 0 and total_episodes_trained % args.evaluate_every == 0:
                    print(f"\n--- Evaluating ALL agents at episode {total_episodes_trained} ---")
                    # Setup evaluation environment (reusable for all evaluations in this step)
                    eval_env = rlcard.make(args.env, config={'seed': args.seed + total_episodes_trained + 10000})

                    # === Evaluation vs Random ===
                    print("  --- Evaluation vs RandomAgent ---")
                    eval_opponents_random = [RandomAgent(num_actions=eval_env.num_actions) for _ in range(1, eval_env.num_players)]

                    # Evaluate DQN Agent vs Random
                    eval_agents_dqn = [agent_dqn] + eval_opponents_random
                    eval_env.set_agents(eval_agents_dqn)
                    perf_dqn_vs_random = tournament(eval_env, args.num_eval_games)[0]
                    multi_logger.log_performance("DQN", total_episodes_trained, perf_dqn_vs_random) # Log using key
                    print(f"    DQN vs Random: {perf_dqn_vs_random}")

                    # Evaluate VAEDQN Agent vs Random
                    eval_agents_vaedqn = [agent_vaedqn] + eval_opponents_random
                    eval_env.set_agents(eval_agents_vaedqn)
                    perf_vaedqn_vs_random = tournament(eval_env, args.num_eval_games)[0]
                    multi_logger.log_performance("VAE_DQN", total_episodes_trained, perf_vaedqn_vs_random) # Log using key
                    print(f"    VAE_DQN vs Random: {perf_vaedqn_vs_random}")

                    # === Mutual Evaluation (DQN vs VAEDQN) ===
                    print("  --- Mutual Evaluation (DQN vs VAE_DQN) ---")
                    if num_players == 2:
                        # Simple 1v1 setup
                        eval_env.set_agents([agent_dqn, agent_vaedqn])
                        payoffs_mutual = tournament(eval_env, args.num_eval_games)
                        print(f"    DQN score (vs VAE_DQN): {payoffs_mutual[0]}")
                        print(f"    VAE_DQN score (vs DQN): {payoffs_mutual[1]}")
                    elif num_players > 2:
                        # Setup with random agents filling remaining slots
                        mutual_eval_opponents = [RandomAgent(num_actions=eval_env.num_actions) for _ in range(num_players - 2)]
                        eval_env.set_agents([agent_dqn, agent_vaedqn] + mutual_eval_opponents)
                        payoffs_mutual = tournament(eval_env, args.num_eval_games)
                        print(f"    DQN score (vs VAE_DQN w/ Randoms): {payoffs_mutual[0]}")
                        print(f"    VAE_DQN score (vs DQN w/ Randoms): {payoffs_mutual[1]}")
                    else: # num_players < 2, should not happen
                         print("    Mutual evaluation requires at least 2 players.")
                    print("-------------------------------------")


                episodes_in_chunk += 1
                total_episodes_trained += 1

            # Switch the learning agent for the next chunk
            current_learner_idx = 1 - current_learner_idx

        # --- Final Steps after main loop (inside 'with MultiLogger') ---
        # Retrieve paths after the loop finishes but before exiting the 'with' block
        # or rely on the get_paths method after the 'with' block exits
        dqn_csv_path, dqn_fig_path = multi_logger.get_paths("DQN")
        vaedqn_csv_path, vaedqn_fig_path = multi_logger.get_paths("VAE_DQN")


    # --- Plotting and Saving (outside 'with MultiLogger') ---
    print(f"\nTraining finished. Plotting learning curves...")
    # Plot DQN curve
    if dqn_csv_path and os.path.exists(dqn_csv_path):
        print(f"  Plotting DQN: {dqn_csv_path} -> {dqn_fig_path}")
        plot_curve(dqn_csv_path, dqn_fig_path, f"DQN_vs_Alternating")
    else:
        print(f"  Warning: CSV file not found for DQN at {dqn_csv_path}, cannot plot.")
    # Plot VAEDQN curve
    if vaedqn_csv_path and os.path.exists(vaedqn_csv_path):
        print(f"  Plotting VAE_DQN: {vaedqn_csv_path} -> {vaedqn_fig_path}")
        plot_curve(vaedqn_csv_path, vaedqn_fig_path, f"VAE_DQN_vs_Alternating")
    else:
        print(f"  Warning: CSV file not found for VAE_DQN at {vaedqn_csv_path}, cannot plot.")


    # Save both trained learning agents
    print("Saving final trained models...")
    save_path_dqn = os.path.join(args.log_dir, 'model_dqn_final.pth')
    save_path_vaedqn = os.path.join(args.log_dir, 'model_vaedqn_final.pth')

    try:
        torch.save(agent_dqn, save_path_dqn)
        print(f'Standard DQN model object saved in {save_path_dqn}')
    except Exception as e:
        print(f"Error saving Standard DQN model object: {e}")

    try:
        torch.save(agent_vaedqn, save_path_vaedqn)
        print(f'VAEDQN model object saved in {save_path_vaedqn}')
    except Exception as e:
        print(f"Error saving VAEDQN model object: {e}")


# ... (Keep if __name__ == '__main__': block as before) ...
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Alternating DQN/VAEDQN Training in RLCard")
    parser.add_argument('--env', type=str, default='bluffgame')
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--num_eval_games', type=int, default=2000)
    parser.add_argument('--evaluate_every', type=int, default=200)
    parser.add_argument('--log_dir', type=str, default='experiments/bluffgame_dqn_vs_vaedqn_mutual_eval_v2/') # Changed default dir

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    # Call the revised multi_train function
    multi_train(args)
