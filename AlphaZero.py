import numpy as np
import random
from collections import deque
import os
import math
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical

BOARD_SIZE = 4
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
BLOCK = 2
inithealth = 2
attackpower = 1
healpower = 1
isDiagHeal = True
isDiagAttack = True

class Game:  
    def __init__(self):
        self.board = np.array([[{'type': EMPTY, 'health': 0} for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)])
        self.currentPlayer = PLAYER_X
        self.boardChanged = False

    def initialize_board(self):
        self.board = np.array([[{'type': EMPTY, 'health': 0} for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)])
        self.currentPlayer = PLAYER_X
        self.boardChanged = False

    def heal_rule(self, row, col):
        currentPlayer = self.board[row][col]['type']
        if currentPlayer == EMPTY or currentPlayer == BLOCK:
            return
        self.board[row][col]['health'] = inithealth
        n = 0
        directNeighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for dr, dc in directNeighbors:
            r, c = row + dr, col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c]['type'] == currentPlayer:
                n += 1
        diagonalNeighbors = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        m = 0
        for dr, dc in diagonalNeighbors:
            r, c = row + dr, col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c]['type'] == currentPlayer:
                m += 1
        if n >= 2:
            if m > 0 and isDiagHeal:
                self.board[row][col]['health'] = inithealth + (n + m) * healpower - 1
            else:
                self.board[row][col]['health'] = inithealth + n * healpower - 1

    def damage_rule(self, row, col, currentPlayer):
        if self.board[row][col]['type'] == EMPTY or self.board[row][col]['type'] == BLOCK:
            return
        opponent = PLAYER_O if currentPlayer == PLAYER_X else PLAYER_X
        n = 0
        directNeighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for dr, dc in directNeighbors:
            r, c = row + dr, col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c]['type'] == opponent:
                n += 1
        diagonalNeighbors = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        m = 0
        for dr, dc in diagonalNeighbors:
            r, c = row + dr, col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c]['type'] == opponent:
                m += 1
        if n >= 2:
            if m > 0 and isDiagAttack:
                self.board[row][col]['health'] -= (n + m) * attackpower
            else:
                self.board[row][col]['health'] -= n * attackpower

    def death_rule(self, row, col):
        if self.board[row][col]['type'] == EMPTY or self.board[row][col]['type'] == BLOCK:
            return
        if self.board[row][col]['health'] <= 0:
            self.board[row][col]['type'] = PLAYER_O if self.board[row][col]['type'] == PLAYER_X else PLAYER_X
            self.board[row][col]['health'] = 2
            self.boardChanged = True

    def block_rule(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                directNeighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
                xCount, oCount = 0, 0
                for dr, dc in directNeighbors:
                    r, c = i + dr, j + dc
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                        if self.board[r][c]['type'] == PLAYER_X:
                            xCount += 1
                        elif self.board[r][c]['type'] == PLAYER_O:
                            oCount += 1
                if xCount >= 2 and oCount >= 2:
                    self.board[i][j]['type'] = BLOCK
                    self.board[i][j]['health'] = 0

    def refresh_board(self):
        self.boardChanged = False
        if not isDiagHeal or not isDiagAttack or attackpower != healpower:
            self.block_rule()
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] != EMPTY:
                    self.heal_rule(i, j)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == self.currentPlayer:
                    self.damage_rule(i, j, self.currentPlayer)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == self.currentPlayer:
                    self.death_rule(i, j)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] != EMPTY:
                    self.heal_rule(i, j)
        opponent = PLAYER_O if self.currentPlayer == PLAYER_X else PLAYER_X
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == opponent:
                    self.damage_rule(i, j, opponent)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == opponent:
                    self.death_rule(i, j)

    def is_board_full(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == EMPTY:
                    return False
        return True

    def count_pieces(self):
        xCount, oCount = 0, 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == PLAYER_X:
                    xCount += 1
                elif self.board[i][j]['type'] == PLAYER_O:
                    oCount += 1
        return xCount, oCount

    def determine_winner(self):
        if self.is_board_full():
            xCount, oCount = self.count_pieces()
            if xCount > oCount:
                return PLAYER_X
            elif oCount > xCount:
                return PLAYER_O
            else:
                return 'Draw'
        return None

    def get_state(self):
        state = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                cell = self.board[i][j]
                if cell['type'] == PLAYER_X:
                    state.append(1)
                elif cell['type'] == PLAYER_O:
                    state.append(-1)
                elif cell['type'] == BLOCK:
                    state.append(0)
                else:
                    state.append(0)
        return np.array(state)

    def get_valid_moves(self):
        valid_moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == EMPTY:
                    valid_moves.append((i, j))
        return valid_moves

    def make_move(self, row, col):
        if self.board[row][col]['type'] == EMPTY:
            self.board[row][col]['type'] = self.currentPlayer
            self.board[row][col]['health'] = inithealth
            self.boardChanged = True
            while self.boardChanged:
                self.boardChanged = False
                self.refresh_board()
            winner = self.determine_winner()
            if winner is not None:
                return winner
            self.currentPlayer = PLAYER_O if self.currentPlayer == PLAYER_X else PLAYER_X
            return None
        return None

class MCTSNode:
    def __init__(self, state, parent=None, prior_prob=0):
        self.state = state  
        self.parent = parent  
        self.children = {}  
        self.visit_count = 0  
        self.total_value = 0  
        self.prior_prob = prior_prob  
        self.u = 0

    def is_fully_expanded(self):
        
        return len(self.children) == len(self.get_legal_actions())

    def get_legal_actions(self):
        
        game = Game()
        
        game.board = np.empty((BOARD_SIZE, BOARD_SIZE), dtype=object)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                game.board[i, j] = {'type': EMPTY, 'health': 0}
        state_index = 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.state[state_index] == 1:
                    game.board[i][j]['type'] = PLAYER_X
                    game.board[i][j]['health'] = inithealth
                elif self.state[state_index] == -1:
                    game.board[i][j]['type'] = PLAYER_O
                    game.board[i][j]['health'] = inithealth
                elif self.state[state_index] == 0:
                    
                    is_block = True
                    directNeighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
                    for dr, dc in directNeighbors:
                        r, c = i + dr, j + dc
                        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                            neighbor_index = r * BOARD_SIZE + c
                            if self.state[neighbor_index] != 0:
                                is_block = False
                                break
                    if is_block:
                        game.board[i][j]['type'] = BLOCK
                        game.board[i][j]['health'] = 0
                    else:
                        game.board[i][j]['type'] = EMPTY
                        game.board[i][j]['health'] = 0
                state_index += 1
        game.currentPlayer = PLAYER_X if self.state.sum() >= 0 else PLAYER_O
        return game.get_valid_moves()

    def select_child(self, c_puct=1.0):
        
        best_child = None
        best_ucb = -float('inf')

        children_list = list(self.children.items())
        if not children_list:
            return None

        for action, child in children_list:
            
            ucb = child.total_value / (child.visit_count + 1e-8) + \
                  c_puct * child.prior_prob * math.sqrt(self.visit_count) / (1 + child.visit_count)
            child.u = ucb

            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        if best_child is None:
            
            return random.choice(children_list)[1]
        
        return best_child

    def expand(self, action_probs):
        
        for action, prob in action_probs.items():
            if action not in self.children:
                
                game = Game()
                game.board = np.empty((BOARD_SIZE, BOARD_SIZE), dtype=object)
                for i in range(BOARD_SIZE):
                    for j in range(BOARD_SIZE):
                        game.board[i, j] = {'type': EMPTY, 'health': 0}
                state_index = 0
                for i in range(BOARD_SIZE):
                    for j in range(BOARD_SIZE):
                        if self.state[state_index] == 1:
                            game.board[i][j]['type'] = PLAYER_X
                            game.board[i][j]['health'] = inithealth
                        elif self.state[state_index] == -1:
                            game.board[i][j]['type'] = PLAYER_O
                            game.board[i][j]['health'] = inithealth
                        elif self.state[state_index] == 0:
                            
                            is_block = True
                            directNeighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
                            for dr, dc in directNeighbors:
                                r, c = i + dr, j + dc
                                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                                    neighbor_index = r * BOARD_SIZE + c
                                    if self.state[neighbor_index] != 0:
                                        is_block = False
                                        break
                            if is_block:
                                game.board[i][j]['type'] = BLOCK
                                game.board[i][j]['health'] = 0
                            else:
                                game.board[i][j]['type'] = EMPTY
                                game.board[i][j]['health'] = 0
                        state_index += 1
                game.currentPlayer = PLAYER_X if self.state.sum() >= 0 else PLAYER_O
                game.make_move(action[0], action[1])
                new_state = game.get_state()

                
                self.children[action] = MCTSNode(new_state, parent=self, prior_prob=prob)

    def update(self, value):
        
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            self.parent.update(-value)

    def __str__(self):
        return f"State: {self.state.reshape(BOARD_SIZE, BOARD_SIZE)}, Visits: {self.visit_count}, Value: {self.total_value / (self.visit_count + 1e-8)}, Prior Prob: {self.prior_prob}, U: {self.u}"

class AlphaZero:
    def __init__(self, model_path=None, learning_rate=0.001, c_puct=1.0, num_simulations=100, temperature=1.0):
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.temperature = temperature if temperature > 0 else 1e-8 
        self.model = self._build_model() if model_path is None else self.load_model(model_path)

    def _build_model(self):
        
        input_layer = Input(shape=(BOARD_SIZE * BOARD_SIZE,))
        
        
        policy_hidden = Dense(64, activation='relu')(input_layer)
        policy_output = Dense(BOARD_SIZE * BOARD_SIZE, activation='softmax', name='policy')(policy_hidden)

        
        value_hidden = Dense(32, activation='relu')(input_layer)
        value_output = Dense(1, activation='tanh', name='value')(value_hidden)

        model = Model(inputs=input_layer, outputs=[policy_output, value_output])
        model.compile(loss={'policy': 'categorical_crossentropy', 'value': 'mse'},
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def save_model(self, filepath):
        
        self.model.save(filepath)

    def load_model(self, filepath):
        
        return load_model(filepath)

    def mcts(self, root_state):
        
        root = MCTSNode(root_state)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            
            while node is not None and node.is_fully_expanded():
                node = node.select_child(self.c_puct)
                if node is not None:
                    search_path.append(node)

            
            if node is None:
                value = self.expand_and_evaluate(search_path[-1])
            else:
                value = self.expand_and_evaluate(node)

            
            for node in reversed(search_path):
                node.update(value)

        return root

    def expand_and_evaluate(self, node):
      
      game = Game()
      
      game.board = np.empty((BOARD_SIZE, BOARD_SIZE), dtype=object)
      for i in range(BOARD_SIZE):
          for j in range(BOARD_SIZE):
              game.board[i, j] = {'type': EMPTY, 'health': 0}
      state_index = 0
      for i in range(BOARD_SIZE):
          for j in range(BOARD_SIZE):
              if node.state[state_index] == 1:
                  game.board[i][j]['type'] = PLAYER_X
                  game.board[i][j]['health'] = inithealth
              elif node.state[state_index] == -1:
                  game.board[i][j]['type'] = PLAYER_O
                  game.board[i][j]['health'] = inithealth
              elif node.state[state_index] == 0:
                  
                  is_block = True
                  directNeighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
                  for dr, dc in directNeighbors:
                      r, c = i + dr, j + dc
                      if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                          neighbor_index = r * BOARD_SIZE + c
                          if node.state[neighbor_index] != 0:
                              is_block = False
                              break
                  if is_block:
                      game.board[i][j]['type'] = BLOCK
                      game.board[i][j]['health'] = 0
                  else:
                      game.board[i][j]['type'] = EMPTY
                      game.board[i][j]['health'] = 0
              state_index += 1
      
      game.currentPlayer = PLAYER_X if node.state.sum() >= 0 else PLAYER_O
      winner = game.determine_winner()

      if winner is not None:
          
          value = 1 if winner == game.currentPlayer else -1
          return value

      
      policy, value = self.model.predict(node.state.reshape(1, -1), verbose=0)
      policy = policy[0]
      value = value[0][0]

      
      legal_actions = node.get_legal_actions()
      action_probs = {action: policy[action[0] * BOARD_SIZE + action[1]] for action in legal_actions}

      
      node.expand(action_probs)

      return value

      if winner is not None:
          
          value = 1 if winner == game.currentPlayer else -1
          return value

      
      policy, value = self.model.predict(node.state.reshape(1, -1), verbose=0)
      policy = policy[0]
      value = value[0][0]

      
      legal_actions = node.get_legal_actions()
      action_probs = {action: policy[action[0] * BOARD_SIZE + action[1]] for action in legal_actions}

      
      node.expand(action_probs)

      return value

    def get_action_probs(self, state):
        
        root = self.mcts(state)
        game1 = Game()
        
        
        

        
        visit_counts = np.array([root.children.get(action, MCTSNode(None)).visit_count for action in [(i,j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]])
        
        if self.temperature == 0:
            
            action_probs = np.zeros(BOARD_SIZE * BOARD_SIZE)
            action_probs[np.argmax(visit_counts)] = 1
        else:
            
            
            safe_visit_counts = visit_counts + 1e-8
            safe_visit_counts = safe_visit_counts ** (1 / self.temperature)
            action_probs = safe_visit_counts / np.sum(safe_visit_counts)
        
        
        action_probs = action_probs.reshape(BOARD_SIZE, BOARD_SIZE)

        
        valid_moves = game1.get_valid_moves()
        
        masked_probs = np.zeros((BOARD_SIZE, BOARD_SIZE))

        if not valid_moves:
            
            return masked_probs

        for move in valid_moves:
            masked_probs[move[0]][move[1]] = action_probs[move[0]][move[1]]
        
        
        sum_probs = np.sum(masked_probs)
        if sum_probs > 0:
            masked_probs /= sum_probs
        else:
            
            if valid_moves:
                prob = 1.0 / len(valid_moves)
                for move in valid_moves:
                    masked_probs[move[0]][move[1]] = prob
            else:
                print("Warning: No valid moves and sum_probs is zero.")

        return masked_probs

    def self_play(self, num_games=1):
      
      training_data = []

      for _ in range(num_games):
          game = Game()
          game.initialize_board()
          state = game.get_state()
          states, policy_targets, value_targets = [], [], []

          while True:
              
              action_probs = self.get_action_probs(state)

              
              action = self.choose_action(action_probs)

              
              states.append(state)
              policy_target = np.zeros(BOARD_SIZE * BOARD_SIZE)
              policy_target[action[0] * BOARD_SIZE + action[1]] = 1
              policy_targets.append(policy_target)

              
              winner = game.make_move(action[0], action[1])
              
              
              next_state = game.get_state()

              if winner is not None:
                  
                  value = 1 if winner == game.currentPlayer else -1
                  value_targets.extend([value] * len(states))
                  
                  
                  flipped_states = [np.where(s == 1, -1, np.where(s == -1, 1, s)) for s in states]
                  flipped_policy_targets = [np.flip(pt) for pt in policy_targets]
                  
                  
                  training_data.extend(list(zip(flipped_states, flipped_policy_targets, [-v for v in value_targets])))
                  break

              
              state = next_state

          training_data.extend(list(zip(states, policy_targets, value_targets)))

      return training_data

    def choose_action(self, action_probs):
        
        
        actions = np.arange(BOARD_SIZE * BOARD_SIZE)

        
        probs = action_probs.flatten()

        if np.all(probs == 0):
            
            print("Warning: All probabilities are zero. Choosing a random action.")
            chosen_index = np.random.choice(actions)
        elif np.isnan(probs).any():
            print("Warning: NaN detected in probabilities. Choosing a random action.")
            chosen_index = np.random.choice(actions)
        else:
            probs = probs / np.sum(probs)  
            chosen_index = np.random.choice(actions, p=probs)

        return chosen_index // BOARD_SIZE, chosen_index % BOARD_SIZE

    def train(self, training_data, epochs=10, batch_size=32):
        
        states, policy_targets, value_targets = zip(*training_data)

        
        states = np.array(states)
        policy_targets = np.array(policy_targets)
        value_targets = np.array(value_targets)

        
        self.model.fit(states, {'policy': policy_targets, 'value': value_targets},
                       epochs=epochs, batch_size=batch_size, verbose=1)

    def play_against_human(self, human_starts=True):
        
        game = Game()
        game.initialize_board()

        while True:
            print("Current Board:")
            for row in game.board:
                print([cell['type'] for cell in row])

            if (game.currentPlayer == PLAYER_X and human_starts) or \
               (game.currentPlayer == PLAYER_O and not human_starts):
                
                try:
                    row, col = map(int, input("Enter your move (row col): ").split())
                    if (row, col) not in game.get_valid_moves():
                        print("Invalid move! Try again.")
                        continue
                except ValueError:
                    print("Invalid input! Please enter two numbers separated by a space.")
                    continue
                winner = game.make_move(row, col)
            else:
                
                state = game.get_state()
                action_probs = self.get_action_probs(state)
                action = np.unravel_index(np.argmax(action_probs), action_probs.shape)
                print(f"AI moves to: ({action[0]}, {action[1]})")
                winner = game.make_move(action[0], action[1])

            
            if winner is not None:
                print("Final Board:")
                for row in game.board:
                    print([cell['type'] for cell in row])
                if winner == 'Draw':
                    print("It's a draw!")
                else:
                    print(f"Winner: {winner}")
                break


if __name__ == "__main__":
    print("1. Train AlphaZero from scratch")
    print("2. Continue training AlphaZero")
    print("3. Play against AlphaZero")
    choice = "1"

    if choice == "1":
        num_games = 10
        num_iterations = 10
        model_save_path = './res'
        
        
        alphazero = AlphaZero()
        for i in range(num_iterations):
            print(f"Iteration: {i+1}/{num_iterations}")
            training_data = alphazero.self_play(num_games)
            alphazero.train(training_data)
            alphazero.save_model(model_save_path + f"_iteration_{i+1}.keras")

        alphazero.save_model("./res.keras")

    elif choice == "2":
        num_games = int(input("Enter number of self-play games for each iteration: "))
        num_iterations = int(input("Enter number of training iterations: "))
        model_load_path = input("Enter model load path (e.g., models/alphazero_model_iteration_5): ")
        model_save_path = input("Enter model save path (e.g., models/alphazero_model): ")
        
        
        alphazero = AlphaZero(model_path=model_load_path)
        for i in range(num_iterations):
            print(f"Iteration: {i+1}/{num_iterations}")
            training_data = alphazero.self_play(num_games)
            alphazero.train(training_data)
            alphazero.save_model(model_save_path + f"_iteration_{i+1}")

    elif choice == "3":
        model_load_path = input("Enter model load path (e.g., models/alphazero_model_iteration_10): ")
        human_starts = input("Do you want to start first? (y/n): ").lower() == 'y'
        
        
        alphazero = AlphaZero(model_path=model_load_path)
        alphazero.play_against_human(human_starts=human_starts)

    else:
        print("Invalid choice!")



