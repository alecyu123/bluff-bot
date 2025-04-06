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
├── experiments        <- Exploration of RLCard, agent architectures (DQN, VAEDQN), and experiment results
│   ├── blackjack          <- Blackjack game and agent exploration
│   ├── bluffgame          <- DQN, VAEDQN, LSTMVAEDQN experiments and evaluation against random agents, human agents, each other
│   ├── leduc              <- Leduc game and agent exploration
│   └── limit              <- Texas Limit game and agent exploration
│
├── mod-RLCard         <- Modified RLCard environments and agents (please add these to the correct folders in your local RLCard Library)
│   ├── agents             <- Custom agent implementations (e.g., modified DQN (VAEDQN))
|   ├── env                <- environment wrapper around bluffgame needed for RLCard lib internal workings
│   └── games
│       └── bluffgame      <- Custom game bluffgame implementations
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
|   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── references     <- Research papers referenced in reports  
│
├── .gitignore         <- Specifies files and directories to ignore in version control
│
├── LICENSE            <- Open-source license if one is chosen
│
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
│
├── README.md          <- The top-level README for developers using this project.
│
├── TODO.md            <- Task list and project planning notes
│
├── pyproject.toml     <- Project configuration file with package metadata and tool configs
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment (includes RLCard)
│
├── setup.cfg          <- Configuration file for flake8 and other tools



```

--------

