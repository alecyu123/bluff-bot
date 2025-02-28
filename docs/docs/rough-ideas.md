## Brainstorm Session 1

# Think about how the game will work
1. Use a while loop for turns (deterministic) (It's the environment tabulating things to feed to our NNs.)
2. Use OpenAI gym, RLCard, etc. (look into these libraries and their documentation)
3. Think about what the adversarial agents/opponents will take their actions

# State space features (what should the state space represent)?
1. Current Hand Strength
2. Gaussian (representations) --> contains a lot of features within this gaussian feature
    - Unpredictability
    - Conservative/Agressiveness
    - 
3. Pot Size
4. Savings of each player

# Where will need the Neural Networks?
1. Neural network to predict the q-value based on the state/features (past history)
2. Neural network to go from state to state based on action taken in previous state


# ToDo for Next Time
1. Look into OpenSpiel (RL game framework from Deepmind)