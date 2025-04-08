# -*- coding: utf-8 -*-
import re
import ast
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Import seaborn
import pandas as pd # Import pandas

# Paste your multi-line string data here
data = """
0 experiments/bluffgame_dqn_vs_vaedqn_mutual_eval_v5/model_vaedqn_final.pth 1.4654
1 experiments/bluffgame_dqn_vs_vaedqn_mutual_eval_v5/model_dqn_final.pth -1.4654
--- Overall Action Distributions ---
Player 0: {'raise': 22008, 'call': 11994, 'fold': 2076, 'check': 996}
Player 1: {'raise': 28838, 'call': 5616, 'check': 484}

--- First Move Action Distributions (When Player X Moves First) ---
Player 0 (when first): {'raise': 5040}
Player 1 (when first): {'raise': 4454, 'call': 506}

0 experiments/bluffgame_dqn_vs_vaedqn_mutual_eval_v5/model_vaedqn_final.pth 1.44005
1 random -1.44005
--- Overall Action Distributions ---
Player 0: {'raise': 12176, 'check': 1713, 'call': 2756, 'fold': 633}
Player 1: {'fold': 6854, 'call': 5763, 'check': 1218, 'raise': 6962}

--- First Move Action Distributions (When Player X Moves First) ---
Player 0 (when first): {'raise': 5040}
Player 1 (when first): {'call': 1663, 'fold': 1613, 'raise': 1684}

0 experiments/bluffgame_dqn_vs_vaedqn_mutual_eval_v5/model_vaedqn_final.pth 0.7312
1 aggro -0.7312
--- Overall Action Distributions ---
Player 0: {'raise': 27385, 'call': 16911, 'fold': 515, 'check': 1043}
Player 1: {'raise': 31329, 'check': 9485, 'call': 1016}

--- First Move Action Distributions (When Player X Moves First) ---
Player 0 (when first): {'raise': 5040}
Player 1 (when first): {'raise': 4960}

0 experiments/bluffgame_dqn_vs_vaedqn_mutual_eval_v5/model_dqn_final.pth 1.6905
1 random -1.6905
--- Overall Action Distributions ---
Player 0: {'raise': 14615, 'call': 1828, 'check': 1289, 'fold': 59}
Player 1: {'fold': 7473, 'call': 6727, 'raise': 7073, 'check': 891}

--- First Move Action Distributions (When Player X Moves First) ---
Player 0 (when first): {'raise': 4581, 'call': 459}
Player 1 (when first): {'call': 1669, 'fold': 1628, 'raise': 1663}

0 experiments/bluffgame_dqn_vs_vaedqn_mutual_eval_v5/model_dqn_final.pth -0.2484
1 aggro 0.2484
--- Overall Action Distributions ---
Player 0: {'raise': 43142, 'call': 11487}
Player 1: {'raise': 39589, 'call': 8513, 'check': 5419}

--- First Move Action Distributions (When Player X Moves First) ---
Player 0 (when first): {'raise': 4581, 'call': 459}
Player 1 (when first): {'raise': 4960}

0 random -1.529
1 aggro 1.529
--- Overall Action Distributions ---
Player 0: {'fold': 7251, 'call': 5720, 'check': 1757, 'raise': 7123}
Player 1: {'raise': 11695, 'check': 5542, 'call': 388}

--- First Move Action Distributions (When Player X Moves First) ---
Player 0 (when first): {'fold': 1642, 'call': 1709, 'raise': 1689}
Player 1 (when first): {'raise': 4960}
"""

# Helper to get clean agent names
def get_agent_name(raw_name):
    """Simplifies agent identifiers into clean names."""
    if 'vaedqn' in raw_name.lower():
        return 'VAEDQN'
    elif 'dqn' in raw_name.lower():
        print(f"DEBUG: Found 'dqn' in '{raw_name}', assigning name 'DQN'")
        return 'DQN'
    elif 'random' in raw_name.lower():
        return 'Random'
    elif 'aggro' in raw_name.lower():
        return 'Aggro'
    print(f"DEBUG: Raw name '{raw_name}' did not match known patterns, using raw name.")
    return raw_name # Fallback if name doesn't match known patterns

# --- Data Parsing ---
agent_payoffs = defaultdict(list)
agent_action_counts = defaultdict(Counter)

# Regex patterns to find the relevant lines
pattern_p0_info = re.compile(r"^0\s+(.+?)\s+(-?\d+\.?\d*)$")
pattern_p1_info = re.compile(r"^1\s+(.+?)\s+(-?\d+\.?\d*)$")
pattern_p0_actions = re.compile(r"Player 0: ({.*?})$")
pattern_p1_actions = re.compile(r"Player 1: ({.*?})$")

# Split data into blocks for each matchup
blocks = data.strip().split('\n\n') # Assuming blank lines separate matchups

print("--- Parsing Matchups ---")
for i, block in enumerate(blocks):
    lines = block.strip().split('\n')
    agent0_name_raw = None
    agent1_name_raw = None
    payoff0 = None
    payoff1 = None
    actions0_dict = None
    actions1_dict = None

    for line in lines:
        match_p0_info = pattern_p0_info.match(line)
        match_p1_info = pattern_p1_info.match(line)
        match_p0_actions = pattern_p0_actions.search(line)
        match_p1_actions = pattern_p1_actions.search(line)

        if match_p0_info:
            agent0_name_raw = match_p0_info.group(1).strip()
            payoff0 = float(match_p0_info.group(2))
        elif match_p1_info:
            agent1_name_raw = match_p1_info.group(1).strip()
            payoff1 = float(match_p1_info.group(2))
        elif match_p0_actions:
            try:
                actions0_dict = ast.literal_eval(match_p0_actions.group(1))
            except Exception as e:
                print(f"  Warning: Could not parse Player 0 actions: {e} in block {i+1}")
        elif match_p1_actions:
             try:
                actions1_dict = ast.literal_eval(match_p1_actions.group(1))
             except Exception as e:
                print(f"  Warning: Could not parse Player 1 actions: {e} in block {i+1}")

    # Process the extracted info for this block if valid
    if all([agent0_name_raw, agent1_name_raw, payoff0 is not None, payoff1 is not None, 
            actions0_dict, actions1_dict]):
        agent0_name = get_agent_name(agent0_name_raw)
        agent1_name = get_agent_name(agent1_name_raw)
        print(f"  Processed Block {i+1}: Agent0='{agent0_name}', Agent1='{agent1_name}'")
        if 'DQN' in [agent0_name, agent1_name]:
             print(f"    *** DQN detected in this matchup! ***")

        # Aggregate payoffs
        agent_payoffs[agent0_name].append(payoff0)
        agent_payoffs[agent1_name].append(payoff1)

        # Aggregate actions (update Counters)
        agent_action_counts[agent0_name].update(actions0_dict)
        agent_action_counts[agent1_name].update(actions1_dict)
    else:
         print(f"  Warning: Incomplete data found in block {i+1}.")
print("--- Parsing Complete ---")

# --- Data Preparation and Calculation ---

# Calculate average payoffs
avg_payoffs = {
    agent: sum(payoffs) / len(payoffs) if payoffs else 0
    for agent, payoffs in agent_payoffs.items()
}

# Prepare data for plotting
# Get agents present in the action counts (should include all agents with data)
agents = sorted(list(agent_action_counts.keys()))
print(f"\nDEBUG: Agents found after parsing: {agents}")
if 'DQN' not in agents and len(agent_payoffs.get('DQN', [])) > 0:
    print("\n*** CRITICAL DEBUG: 'DQN' has payoff data but is NOT in action count keys! Check parsing. ***\n")
elif 'DQN' not in agents:
    print("\n*** CRITICAL DEBUG: 'DQN' was not found as an agent key after parsing. Check raw data and get_agent_name function. ***\n")


if not agents:
    print("Error: No agent data found after parsing. Exiting.")
    exit()

# Get all unique actions observed across all agents
all_actions = sorted(list(set(action for counts in agent_action_counts.values() for action in counts)))

# Prepare data for individual payoff plots
# Box plot needs a list of lists
payoffs_for_boxplot = [agent_payoffs.get(agent, []) for agent in agents]

# Strip plot (seaborn) works best with a DataFrame in "long format"
payoff_data_list = []
for agent, payoffs in agent_payoffs.items():
    # Ensure we only add data for agents we identified from action counts
    if agent in agents:
        for payoff in payoffs:
            payoff_data_list.append({'Agent': agent, 'Payoff': payoff})
df_payoffs = pd.DataFrame(payoff_data_list)


print("\n--- Processed Summary ---")
print("Agents Found for Plotting:", agents) # Explicitly show agents used for plots
print("Observed Actions:", all_actions)
print("Average Payoffs:", {k: round(v, 4) for k, v in avg_payoffs.items()})
print("Aggregated Action Counts:", {k: dict(v) for k, v in agent_action_counts.items()})


# --- Plotting ---

print("\n--- Starting Plot Generation ---")
plt.style.use('seaborn-v0_8-talk') # Use a visually appealing style

# Define a consistent color map for agents across plots
# Make sure color map generation doesn't crash if agents list is empty
agent_colors = plt.cm.get_cmap('viridis', len(agents)) if agents else None

# 1. Plot Average Payoffs
print("Plotting: Average Payoffs...")
fig1, ax1 = plt.subplots(figsize=(8, 6))
payoff_values = [avg_payoffs.get(agent, 0) for agent in agents]
colors_payoff = ['skyblue' if p >= 0 else 'salmon' for p in payoff_values]

ax1.bar(agents, payoff_values, color=colors_payoff, edgecolor='black')
ax1.set_ylabel('Average Payoff')
ax1.set_title('Average Payoff per Agent (vs Others)')
ax1.axhline(0, color='grey', linewidth=1.0, linestyle='--')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
# Add value labels on bars
for i, v in enumerate(payoff_values):
    ax1.text(i, v + (0.05 if v >= 0 else -0.1), f"{v:.3f}", ha='center', va='bottom' if v >=0 else 'top', fontsize=9)


# 2. Plot Action Distributions (Grouped Bar Chart)
print("Plotting: Action Distributions...")
n_agents = len(agents)
n_actions = len(all_actions)
bar_width = 0.8 / n_agents # Calculate width based on number of agents
index = np.arange(n_actions) # x locations for groups

fig2, ax2 = plt.subplots(figsize=(12, 7))

for i, agent in enumerate(agents):
    print(f"  Plotting actions for: {agent}")
    if agent == 'DQN':
        print(f"    DEBUG: DQN action counts being plotted: {agent_action_counts.get('DQN', Counter())}")

    # Get counts for this agent, ensuring all actions are present (with 0 count if missing)
    counts = [agent_action_counts[agent].get(action, 0) for action in all_actions]
    # Calculate bar positions for this agent within the group
    bar_positions = index + i * bar_width - (bar_width * (n_agents - 1) / 2)
    ax2.bar(bar_positions, counts, bar_width, label=agent, color=agent_colors(i/n_agents), edgecolor='grey')

ax2.set_xlabel('Action Type')
ax2.set_ylabel('Total Count (Log Scale)')
ax2.set_yscale('log')
ax2.set_title('Overall Action Distribution per Agent')
ax2.set_xticks(index)
ax2.set_xticklabels(all_actions)
ax2.legend(title="Agents", bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(axis='y', linestyle='--', alpha=0.6)
fig2.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend


# 3. Box Plot of Individual Payoffs
print("Plotting: Box Plot Payoffs...")
fig3, ax3 = plt.subplots(figsize=(10, 7))

if 'DQN' in agents:
     print(f"  DEBUG: Data for DQN in boxplot: {agent_payoffs.get('DQN', [])}")

bp = ax3.boxplot(
    payoffs_for_boxplot, labels=agents, patch_artist=True, showmeans=True,
    meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"},
    medianprops={"color":"black", "linewidth":1.5})

box_colors = [agent_colors(i/n_agents) for i in range(n_agents)]
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.set_ylabel('Payoff per Matchup')
ax3.set_title('Distribution of Individual Payoffs per Agent')
ax3.axhline(0, color='grey', linewidth=1.0, linestyle='--')
ax3.grid(axis='y', linestyle='--', alpha=0.7)


# 4. Strip Plot of Individual Payoffs (using Seaborn)
print("Plotting: Strip Plot Payoffs...")
fig4, ax4 = plt.subplots(figsize=(10, 7))

if 'DQN' in df_payoffs['Agent'].unique():
    print(f"  DEBUG: DataFrame contains DQN data for stripplot.")
    print(df_payoffs[df_payoffs['Agent'] == 'DQN'].head()) # Show first few rows for DQN
else:
    print("  WARNING: DataFrame for stripplot does NOT contain DQN data!")


sns.stripplot(x='Agent', y='Payoff', data=df_payoffs, ax=ax4,
              order=agents,
              jitter=0.25,
              palette=[agent_colors(i/n_agents) for i in range(n_agents)],
              size=7,
              alpha=0.6)

# Overlay the average payoff
sns.pointplot(x='Agent', y='Payoff', data=df_payoffs, ax=ax4,
              order=agents,
              estimator=np.mean,
              markers='D', color='black',
              linestyles='',
              errorbar=None,
              scale=1.2)

ax4.set_ylabel('Payoff per Matchup')
ax4.set_title('Individual Payoffs per Agent (Each Point is a Matchup, Diamond is Mean)')
ax4.axhline(0, color='grey', linewidth=1.0, linestyle='--')
ax4.grid(axis='y', linestyle='--', alpha=0.7)


# Display all plots
print("\n--- Showing Plots ---")
plt.show()
print("--- Script Finished ---")