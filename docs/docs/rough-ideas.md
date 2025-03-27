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


## Brainstorm Session 2

RL Poker
The key insight is that we’re predicting enemy/user behavior, and only after that prediction is our agent going to make a choice
Also we are training our agent to most efficiently learn the enemy/user behavior using its resources, (e.g.) for music recommendations, we don’t want to explore too much and give users really different songs or they will just stop using the app, similarly in poker you don’t want to bet too much just to understand how the enemy will play

Not just learn player action probability distributions as that changes for every different player, but LEARNING TO LEARN player distribution as efficiently as possible

The act of using our resources efficiently between learning (explore, although it must be said that as described by the paragraph right above, learning/generalizing efficiently (learning to learn/optimizing our agent to learn well) is extremely important) and using the knowledge we learned to bet (exploiting) must be also incorporated into our continuous learning process as the end goal is to win money, learning enemy behaviors is just a step towards that (although it is a major and very important one)

Maybe need a prior as well (past enemy data collection for real world at least)
For NN2, we can update the weights of our rnn/lstm every round. The reward/penalty will be based on money gained if we had purely exploited (even if agent chose to not raise because of exploration bias, if it had ignored the exploration and purely exploited, would what would have happened align with our state space)

Actually should we just reveal enemy hand strength to determine how good our NN2 is, it can definitely help. See how close our prediction was to the enemy hand strength. When should we reveal enemy hand, after every turn, cycle, round? Realize that in the real game, it won’t be able to see the cards. Also, will the model update its weights at game time? Or will the weights remain unchanged after training and the model (weights) will learn to process any combination of states.

The state space x1 … xn should in the end contain the strength of each players hand as an important component. If this is the case we can maybe even train after each cycle or even turn. Maybe even instead of x1 …xn being the exact estimated strength, it should be a range or maybe even a Gaussian with mean and variance, so that NN1 can better predict. Almost like a VAE

This is actually allat and we can view it as a Bayesian optimization! Trying to approximate the real world with a surrogate function!

For NN1 we will also update weight every round. The reward/penalty will both be based on exploitation (money gained or lost in that round and also exploration (knowledge about opponent gained

Possible actions in turn: fold, call, check, raise (x money), 

Game vs round (pot got vs cycle (all players played once) vs turn
Also maybe a good idea to feed into NN1 a deterministic relative hand strength (so we don’t have to bother making our NN train to guess deterministic information). This relative hand strength would be calculated by (e.g. we have 0.5 and there are 2 opponents, then we have 0.50.5 chance of being strongest hand, 0.50.52 chance of being middle, and 0.50.5 chance of being worst. Binomial distribution with win or lose bring outcomes per player.

3 main ideas:
Train NN2 separately so that it learns states which should represent enemy hand strength and our agent confidence about enemy hand strength (Gaussian). Actually these states should also represent “qualities of players” such as their aggressiveness, unpredictability, patience, etc. This general characteristic prediction of enemy players along with the guess of their current hand strength should allow our agent (NN1) to predict the actions the opponents will take, and learn how to play based on the predicted actions. This is important when we want to bluff (e.g our hand strength is 0.2, we predict enemy hand strength to be 0.8, but we learned that they are not agressive, so there is a chance that he might not go all in and we can exploit that characteristic) Also we can probably use multiple neural networks, one for each enemy player
Train NN1 to act not only on how to exploit what it thinks the NN1 hand is (mean of the Gaussian), but also choose actions that will reduce the variance of the Gaussian to make better predictions later.
Train NN1 to also choose actions that make itself unpredictable (sample from a Gaussian of actions, with the mean of the Gaussian being the “best move” and using a variance to choose around it)
Just skim through our thought process so that we’re on the same page
