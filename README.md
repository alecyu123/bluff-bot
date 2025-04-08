# bluff-baby

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

UofT ECE324 Project

## Project Organization

```
├── docs               <- Documentation (brainstorming notes, research, and presentation slides)
│   ├── brainstorming      <- Early planning and state representation pipeline diagrams
│   ├── docs               <- documentation of important project milestones, refined ideas
│   ├── research_papers    <- Useful research paper notes and summaries 
│   └── slideshow          <- Presentation slides for ECE324
│
├── mod-RLCard         <- Modified RLCard Library
│   └── rlcard         <- New RLCard
│
├── simulations        <- Exploration of RLCard, agent architectures (DQN, VAEDQN), and experiment results
│   ├── blackjack          <- Blackjack game and agent exploration
│   ├── bluffgame          <- DQN, VAEDQN, LSTMVAEDQN experiments and evaluation against random agents, human agents, each other
│   ├── leduc              <- Leduc game and agent exploration
│   └── limit              <- Texas Limit game and agent exploration
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── references     <- Research papers referenced in reports  
│
├── .gitignore         <- Specifies files and directories to ignore in version control
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── TODO.md            <- Task list and project planning notes
├── pyproject.toml     <- Project configuration file with package metadata and tool configs
├── requirements.txt   <- The requirements file for reproducing the analysis environment (includes RLCard)
├── setup.cfg          <- Configuration file for flake8 and other tools

```

--------

# Actions Required to Reproduce

1. After pip installing RLCard, run the command pip show rlcard to find the location where you installed the library
2. take the rlcard file inside the mod-RLCard folder, and replace the original rlcard file inside your pip library with this modified version
3. You can now reproduce all experiments inside the experiments folder by just running them (you can change the default arguments).
4. To find all the files changed inside the original rlcard, you can open your new library and look through anything that references bluffgame (see below for new).

# Changed rlcard Directory and files

├───__init__.py
│
├───agents
│   ├───aggressive_agent.py                 <--- new
│   ├───cfr_agent.py
│   ├───dqn_agent.py
│   ├───modified_dqn_agent.py               <--- new
│   ├───nfsp_agent.py
│   ├───pettingzoo_agents.py
│   ├───random_agent.py
│   ├───__init__.py
│   │
│   ├───dmc_agent
│   │   ...
│   │
│   ├───human_agents
│   │   ├───blackjack_human_agent.py
│   │   ├───bluffgame_human_agent.py        <--- new
│   │   ├───leduc_holdem_human_agent.py
│   │   ├───limit_holdem_human_agent.py
│   │   ├───nolimit_holdem_human_agent.py
│   │   ├───uno_human_agent.py
│   │   ├───__init__.py
│   │   │
│   │   ├───gin_rummy_human_agent
│   │   │   ...
│
├───envs
│   ├───blackjack.py
│   ├───bluffgame.py                        <--- new
│   ├───bridge.py
│   ├───doudizhu.py
│   ├───env.py                              <--- changed
│   ├───gin_rummy.py
│   ├───leducholdem.py
│   ├───limitholdem.py
│   ├───mahjong.py
│   ├───nolimitholdem.py
│   ├───registration.py                     <--- changed
│   ├───uno.py
│   ├───__init__.py                         <--- changed
│
├───games
│   ├───base.py
│   ├───__init__.py                         <--- changed
│   │
│   ├───blackjack
│   │   ...
│   │
│   ├───bluffgame                           <--- new
│   │   ├───card2index.json                 <--- new
│   │   ├───dealer.py                       <--- new
│   │   ├───game.py                         <--- new
│   │   ├───judger.py                       <--- new
│   │   ├───player.py                       <--- new
│   │   ├───round.py                        <--- new
│   │   ├───__init__.py                     <--- new
│   │
│   ├───bridge
│   │   ...
│   │
│   ├───doudizhu
│   │   ...
│   │
│   ├───gin_rummy
│   │   ...
│   │
│   ├───leducholdem
│   │   ...
│   │
│   ├───limitholdem
│   │   ...
│   │
│   ├───mahjong
│   │   ...
│   │
│   ├───nolimitholdem
│   │   ...
│   │
│   ├───uno
│   │   ...
│
├───models
│   ├───bridge_rule_models.py
│   ├───doudizhu_rule_models.py
│   ├───gin_rummy_rule_models.py
│   ├───leducholdem_rule_models.py
│   ├───limitholdem_rule_models.py
│   ├───model.py                            <--- new
│   ├───pretrained_models.py
│   ├───registration.py                     <--- new
│   ├───uno_rule_models.py
│   ├───__init__.py                         <--- new
│   │
│   ├───pretrained
│   │   ...
│
├───saved_models
│
├───utils
│   ├───logger.py                           <--- changed
│   ├───pettingzoo_utils.py
│   ├───seeding.py
│   ├───utils.py                            <--- changed
│   ├───__init__.py                         <--- changed