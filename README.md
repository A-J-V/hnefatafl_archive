# Hnefatafl

## Post-Mortem

Despite learning a lot from this project, it ultimately failed in its objective of training a Hnefatafl agent via RL. Here is a laundry list of what went wrong.

I original envisioned this as using AlphaZero-style training. However, I had no real experience in RL when I started it, so picking one of the most complicated RL algorithms as a first attempt was very foolish, and several problems ensued both from my lack of understanding policy-gradient methods as well as design missteps.

First, I decided to imitate the tensor shape of AlphaZero's inputs and outputs in the game environment so that the game board is represented as a 3x11x11 ndarray and the action space as a 40x11x11 ndarray. This choice was, in retrospect, nonsensical. I should have opted for a highly readable data structure to hold the game state, and then applied a transformation when it was time to convert it to tensors for training. Although this seems obvious now, I didn't catch it in time and had baked the hideous ndarray data structures into the game environment when I realized that it was a bad decision.

The second massive misstep is that I decided that I wanted to try something new and see if I could use MCTS to learn the game just with value-estimates. This came back to bite me, because my MCTS design fundamentally assumed that we are not using policy-gradient methods. But when it failed to learn, I switched to policy-gradient methods.

The third misstep is that I used a novel network architecture based on attention-mechanisms. Although I'm convinced it would work well if trained properly, since it has no precedent, I had no reference for good hyperparameters or references to help prevent pitfalls with training such a network in this task. I should have stuck with ResNet50 like AlphaZero chose, or picked an architecture similar to those used in any modern RL-based Chess engine just to have the confidence that the architecture was not a problem.

The fourth misstep is that I tried to use joint-learning. I used the same network for both sides. The problem is that unlike chess, Hnefatafl is asymmetrical--wildly asymmetrical. Each side has a different number of pieces, completely different objectives, and different starting layouts on the board. The attackers have an objectively more complex goal in the game. While the defenders can win simply by moving the King to a corner--an event that can happen due to random chance if the agent is making the King wander around--the attackers must surround the King on 4 sides, which is something far less likely to happen by chance. As a result, attempting to train agents in this game requires handling the fact that an untrained network is going to have a defender victory 95%+ of the time. I eventually tried splitting each side into a different agent with a different network, but once again, since this wasn't my original intention, it didn't quite fit in well to previous code. I committed prematurely to certain design choices and it left the codebase unable to be adapted effectively when needed.

I coded an implementation of PPO and GAE, but the state-space and action-space of Hnefatafl are so huge and the branching factor is so high that it failed to learn even the most basic of tasks. I tried implementing curriculum learning, and it didn't help at all. The last resort would have been to resurrect the MCTS portion of the code, but since I had baked the assumption of using value-based algorithms only, my MCTS code was not compatible with the policy-gradient method.

Overall, while I learned a huge amount about RL throughout this project, I made too many design mistakes and the codebase become a mess of tangled ideas and salvage attempts over the course of several months.

In retrospect I should have:
1) Not chosen the most difficult thing that I could think of for a first RL project,
2) Opted for simple, understandable data structures in the code base, then transformed them as needed in a pipeline when it was necessary for model training,
3) Began with methods that had a clear precedent, like copying AlphaZero's method exactly instead of deviating from it without the understanding to do so intelligently,
4) Recognized the extreme asymmetry of Hnefatafl and created a separate attacker network and defender network from the beginning rather than trying to let both sides share parameters in a joint learning setup,
5) Added extensive unit and integration testing to the RL code. The game logic was tested extensively, but the ML needed more. There may still be bugs hiding that were never found,
6) Written a Tic-Tac-Toe environment to use for sanity testing. It's small enough that the complete game-tree can be viewed and the state-action space is tiny, so it can be used to debug both RL and MCTS during dev.

I don't know if I will have the energy this year to retry this, because there is nothing salvagable in this project except for an outline for the game logic, which as far as I can tell did work (except that I didn't code stalemates).




---------------------------------------------------------------------------------------------------------------------
UPDATE: This project did not succeed, and should not be used as a reference for implementing anything. See Post-Mortem above.

This is a Python implementation of Hnefatafl, aka "Viking Chess."

This is a prototype designed to solidify design choices and train a top quality AI before coding a performant Rust version.

The prototype uses the rules of "Copenhagen Hnefatafl", although the code aims to be modular enough to easily compose variants.

![file-aFooJf2FbEwXEI4HjEbc7UsD](https://github.com/A-J-V/Hnefatafl/assets/72227828/ad82e6ec-d991-456d-a110-d31b98fa20e8)
