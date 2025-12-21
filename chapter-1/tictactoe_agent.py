"""
Tic-Tac-Toe Reinforcement Learning Agent.

This module implements a simple reinforcement learning agent that learns to play
Tic-Tac-Toe using a value-based approach. The agent learns the value of each state
(board configuration) through self-play and play against a random opponent.
"""

import random

class TicTacToe:
    """
    A class representing the Tic-Tac-Toe game environment.
    
    The board is represented as a list of 9 integers:
    0: empty, 1: player, -1: opponent.
    """
    def __init__(self):
        """Initialize the game board and state."""
        self.board = [0] * 9 # 0: empty, 1: player, -1: opponent
        self.is_player_turn = True
        self.game_over = False
    
    def get_state_hash(self):
        """
        Converts the board to a tuple so it can be used as a dict key.
        
        Returns:
            tuple: The current state of the board.
        """
        return tuple(self.board)

    def get_next_state_hash(self, move: int):
        """
        Predicts the state hash after a given move.
        
        Args:
            move (int): The board index where the move is made.
            
        Returns:
            tuple: The state hash of the board after the move.
        """
        new_board = self.board.copy()
        new_board[move] = 1 if self.is_player_turn else -1
        return tuple(new_board)

    def available_moves(self) -> list[int]:
        """
        Returns a list of indices for all empty cells on the board.
        
        Returns:
            list[int]: Indices of available moves.
        """
        return [i for i, x in enumerate(self.board) if x == 0]
    
    def check_winner(self):
        """
        Checks if there is a winner or a draw.
        
        Returns:
            int or None: 1 if player wins, -1 if opponent wins, 0 for draw,
                         and None if the game is still ongoing.
        """
        wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for i, j, k in wins:
            if self.board[i] == self.board[j] == self.board[k] != 0:
                return self.board[i]
        
        if 0 not in self.board: # draw
            return 0
        
        return None # game not over
    
    def step(self, position):
        """
        Executes a move at the specified position.
        
        Args:
            position (int): The board index to place the current player's mark.
            
        Returns:
            int or None: The result of the game after the move.
        """
        if self.board[position] != 0:
            return None

        self.board[position] = 1 if self.is_player_turn else -1
        self.is_player_turn = not self.is_player_turn
        
        result = self.check_winner()
        if result is not None:
            self.game_over = True
        
        return result

class Agent:
    """
    A reinforcement learning agent that learns the value of Tic-Tac-Toe states.
    
    Attributes:
        values (dict): A mapping from state hashes to their learned values.
        step_size (float): The learning rate (alpha).
        epsilon (float): The exploration rate for the epsilon-greedy policy.
        history (list): A sequence of state hashes encountered during a game.
    """
    def __init__(self, step_size=0.1, epsilon=0.1):
        """
        Initialize the agent with learning parameters.
        
        Args:
            step_size (float): Learning rate.
            epsilon (float): Exploration rate.
        """
        self.values = {}
        self.step_size = step_size
        self.epsilon = epsilon
        self.history = [] # store the sequence of states
    
    def get_value(self, state_hash):
        """
        Retrieves the value of a state from the value table.
        
        Args:
            state_hash (tuple): The state to look up.
            
        Returns:
            float: The estimated value of the state.
        """
        if state_hash not in self.values:
            return 0.3 # default initial value
        return self.values[state_hash]
    
    def choose_action(self, env: TicTacToe):
        """
        Chooses an action using an epsilon-greedy policy.
        
        Args:
            env (TicTacToe): The current game environment.
            
        Returns:
            int: The chosen board index for the move.
        """
        moves = env.available_moves()
        # explore
        if random.random() < self.epsilon:
            return random.choice(moves)

        # exploit
        values = [self.get_value(env.get_next_state_hash(move)) for move in moves]
        return moves[values.index(max(values))]
        
    def update_values(self, final_reward: int):
        """
        Updates the value table using the sequence of states from the last game.
        
        Uses the temporal difference update rule: V(S) = V(S) + alpha * (target - V(S)).
        Updates are performed in reverse order from the end of the game.
        
        Args:
            final_reward (int): The reward received at the end of the game.
        """
        # update the value table working backwards from the end of the game
        target = final_reward
        for state_hash in self.history[::-1]:
            value = self.get_value(state_hash)
            # update rule: V(S) = V(S) + alpha * (target - V(S))
            self.values[state_hash] = value + self.step_size * (target - value)
            target = self.values[state_hash] # value of the next state becomes the target for the current state
        self.history = [] # clear the history

def simulate():
    """
    Trains the agent by simulating multiple games against a random opponent.
    
    The simulation runs for a fixed number of games, updating the agent's 
    value table after each game. Finally, it prints the number of learned 
    states and the value of the initial (empty) board state.
    """
    agent = Agent(step_size=0.1, epsilon=0.1)
    n_games = 10000

    print("Training...")

    for i in range(n_games):
        env = TicTacToe()
        
        while not env.game_over:
            # agent turn
            current_state = env.get_state_hash()
            agent.history.append(current_state)
            move = agent.choose_action(env)
            env.step(move)

            if env.game_over:
                # reward: 1 for win, 0 for draw/loss
                reward = 1 if env.check_winner() == 1 else 0
                agent.update_values(reward)
                break

            # opponent turn
            moves = env.available_moves()
            move = random.choice(moves)
            env.step(move)
            
            if env.game_over:
                agent.update_values(0)
                
    print(f"Training complete. Agent has learned values for {len(agent.values)} states.")
    print(f"Value of empty board (start state): {agent.get_value(tuple([0] * 9)):.4f}")

if __name__ == "__main__":
    simulate()