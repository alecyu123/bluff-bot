import os
import argparse
import torch
from collections import Counter, defaultdict

import rlcard
from rlcard.agents import DQNAgent, RandomAgent, VAEDQNAgent
from rlcard.utils import get_device, set_seed, tournament_with_traj

def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        # Using weights_only=False bypasses safe loading and loads the full checkpoint.
        # CAUTION: Only use this with checkpoints from trusted sources.
        print(device)
        agent = torch.load(model_path, map_location=device, weights_only=False)
        agent.set_device(device)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    elif model_path == 'aggro':
        from rlcard.agents import AggressiveAgent
        agent = AggressiveAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]

    return agent

def analyze_action_distributions(list_of_game_logs):
    """
    Analyzes game logs for:
    1. Overall action distributions per player.
    2. Distribution of actions taken by Player 0 specifically when they make
       the absolute first move of the game.
    3. Distribution of actions taken by Player 1 specifically when they make
       the absolute first move of the game.

    Args:
        list_of_game_logs: A list where each element is a list of tuples,
                           representing a game's action sequence.
                           Each tuple is (player_id, action_string).
                           Example: [
                               [(0, 'call'), (1, 'check'), (0, 'raise'), ...],
                               [(0, 'raise'), (1, 'call'), (1, 'fold'), ...],
                               ...
                           ]

    Returns:
        A dictionary containing four Counter objects:
        - 'player0_overall': Overall action distribution for player 0 across all turns.
        - 'player1_overall': Overall action distribution for player 1 across all turns.
        - 'player0_when_first': Distribution of Player 0's actions ONLY when
                                 they made the absolute first move of the game.
        - 'player1_when_first': Distribution of Player 1's actions ONLY when
                                 they made the absolute first move of the game.
    """

    # Initialize Counters for overall distributions
    player0_overall_counts = Counter()
    player1_overall_counts = Counter()

    # Initialize Counters for first action distributions *specific to the player*
    player0_first_move_action_counts = Counter()
    player1_first_move_action_counts = Counter()

    # Iterate through each game log in the list
    for game_log in list_of_game_logs:
        # Skip empty game logs if they exist
        if not game_log:
            continue

        # --- Identify the Absolute First Move and the Player Who Made It ---
        first_player_id, first_action = game_log[0]

        # --- Count the First Move Action for the Specific Player ---
        if first_player_id == 0:
            player0_first_move_action_counts[first_action] += 1
        elif first_player_id == 1:
            player1_first_move_action_counts[first_action] += 1
        else:
            # Optional: Handle unexpected first player ID if necessary
            print(f"Warning: Encountered unexpected player ID in first move: {first_player_id}")

        # --- Count Overall Actions (iterate through the whole log) ---
        for player_id, action in game_log:
            if player_id == 0:
                player0_overall_counts[action] += 1
            elif player_id == 1:
                player1_overall_counts[action] += 1
            else:
                # Handle unexpected player IDs if necessary
                print(f"Warning: Encountered unexpected player ID during overall count: {player_id}")
                continue # Skip this action

    return {
        'player0_overall': player0_overall_counts,
        'player1_overall': player1_overall_counts,
        # Use descriptive keys for the specific first-move distributions
        'player0_when_first': player0_first_move_action_counts,
        'player1_when_first': player1_first_move_action_counts,
    }

def evaluate(args):

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    rewards, trajectory = tournament_with_traj(env, args.num_games)
    for position, reward in enumerate(rewards):
        print(position, args.models[position], reward)
    
    return trajectory
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='bluffgame',
        choices=[
            'bluffgame',
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
        ],
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=[
            'random', 
            'aggro',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=10000, # change
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    game_distributions = evaluate(args)
    distributions = analyze_action_distributions(game_distributions)

    print("--- Overall Action Distributions ---")
    print("Player 0:", dict(distributions['player0_overall']))
    print("Player 1:", dict(distributions['player1_overall']))
    print("\n--- First Move Action Distributions (When Player X Moves First) ---")
    print("Player 0 (when first):", dict(distributions['player0_when_first']))
    print("Player 1 (when first):", dict(distributions['player1_when_first']))
