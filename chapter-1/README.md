# Tic-Tac-Toe Reinforcement Learning

This repository contains a simple reinforcement learning agent that learns to play Tic-Tac-Toe through self-play and play against a random opponent.

## Project Overview

The project consists of two main components:
- **`TicTacToe` Environment**: Simulates the game rules, board state, and player turns.
- **`Agent`**: A reinforcement learning agent that maintains a value table for board states. It uses an epsilon-greedy strategy for exploration and updates its values based on the game outcome using temporal difference learning.

### Running the Simulation
To train the agent and see the learned value of the starting state, run:

```bash
python3 tictactoe_agent.py
```

## How it Works

The agent learns by playing 10,000 games. During each game:
1. It looks ahead one step at all possible moves.
2. It chooses the move that leads to the state with the highest learned value (exploitation), with a small probability of choosing a random move (exploration).
3. At the end of the game, it receives a reward:
   - **Win**: 1.0
   - **Draw/Loss**: 0.0
4. It then updates the values of all states visited during the game using the update rule:
   `V(S) = V(S) + alpha * [V(S') - V(S)]`
   where `alpha` is the learning rate and `V(S')` is the value of the next state (or the final reward).

