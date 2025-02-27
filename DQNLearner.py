import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

BOARD_SIZE = 6
EMPTY = '-'
PLAYER_X = 'x'
PLAYER_O = 'o'
BLOCK = '☒'
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

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        
        
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        
        
        act_values = self.model.predict(state)  
        
        
        if len(act_values.shape) == 1:
            act_values = np.expand_dims(act_values, axis=0)
        
        
        valid_actions = [i * BOARD_SIZE + j for (i, j) in valid_moves]
        valid_act_values = [act_values[0][i] for i in valid_actions]  
        
        
        return valid_moves[np.argmax(valid_act_values)]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                
                next_state_prediction = self.model.predict(next_state)
                
                if len(next_state_prediction.shape) == 1:
                    next_state_prediction = np.expand_dims(next_state_prediction, axis=0)
                target = reward + self.gamma * np.amax(next_state_prediction[0])

            
            if len(state.shape) == 1:
                state = np.expand_dims(state, axis=0)

            
            target_f = self.model.predict(state)
            
            if len(target_f.shape) == 1:
                target_f = np.expand_dims(target_f, axis=0)

            
            row, col = action
            action_index = row * BOARD_SIZE + col

            
            target_f[0][action_index] = target

            
            self.model.fit(state, target_f, epochs=1, verbose=0)

        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def train_ai(episodes=1000, batch_size=32, model_save_path="ai_model.h5"):
    state_size = BOARD_SIZE * BOARD_SIZE
    action_size = BOARD_SIZE * BOARD_SIZE
    agent = DQNAgent(state_size, action_size)
    game = Game()
    for e in range(episodes):
        game.initialize_board()
        state = game.get_state()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            valid_moves = game.get_valid_moves()
            action = agent.act(state, valid_moves)
            row, col = action
            winner = game.make_move(row, col)
            next_state = game.get_state()
            next_state = np.reshape(next_state, [1, state_size])
            reward = 0
            if winner == PLAYER_X:
                reward = 1
            elif winner == PLAYER_O:
                reward = -1
            elif winner == 'Draw':
                reward = 0
            done = winner is not None
            agent.remember(state, action, reward, next_state, done)
            state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 100 == 0:
            print(f"Episode: {e}/{episodes}, Epsilon: {agent.epsilon}")
    
    
    import os
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    agent.save(model_save_path)  
    print(f"Model saved to: {model_save_path}")

def continue_training_ai(episodes=1000, batch_size=32, model_load_path="ai_model.h5", model_save_path="ai_model_continued.h5"):
    state_size = BOARD_SIZE * BOARD_SIZE
    action_size = BOARD_SIZE * BOARD_SIZE
    agent = DQNAgent(state_size, action_size)
    agent.load(model_load_path)  
    game = Game()
    for e in range(episodes):
        game.initialize_board()
        state = game.get_state()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            valid_moves = game.get_valid_moves()
            action = agent.act(state, valid_moves)
            row, col = action
            winner = game.make_move(row, col)
            next_state = game.get_state()
            next_state = np.reshape(next_state, [1, state_size])
            reward = 0
            if winner == PLAYER_X:
                reward = 1
            elif winner == PLAYER_O:
                reward = -1
            elif winner == 'Draw':
                reward = 0
            done = winner is not None
            agent.remember(state, action, reward, next_state, done)
            state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 100 == 0:
            print(f"Episode: {e}/{episodes}, Epsilon: {agent.epsilon}")
    
    
    import os
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    agent.save(model_save_path)  
    print(f"Model saved to: {model_save_path}")

def start_game_with_ai(model_load_path="ai_model.h5"):
    state_size = BOARD_SIZE * BOARD_SIZE
    action_size = BOARD_SIZE * BOARD_SIZE
    agent = DQNAgent(state_size, action_size)
    agent.load(model_load_path)  
    game = Game()
    game.initialize_board()
    while True:
        
        print("Current Board:")
        for row in game.board:
            print([cell['type'] for cell in row])
        
        if game.currentPlayer == PLAYER_X:
            
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
            state = np.reshape(state, [1, state_size])
            valid_moves = game.get_valid_moves()
            action = agent.act(state, valid_moves)
            row, col = action
            print(f"AI moves to: ({row}, {col})")
            winner = game.make_move(row, col)
        
        
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
    print("1. Train AI from scratch")
    print("2. Continue training AI")
    print("3. Play against AI")
    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        episodes = int(input("Enter number of episodes to train: "))
        model_save_path = input("Enter model save path (e.g., models/ai_model.h5): ")
        train_ai(episodes=episodes, model_save_path=model_save_path)
    elif choice == "2":
        episodes = int(input("Enter number of episodes to continue training: "))
        model_load_path = input("Enter model load path (e.g., models/ai_model.h5): ")
        model_save_path = input("Enter model save path (e.g., models/ai_model_continued.h5): ")
        continue_training_ai(episodes=episodes, model_load_path=model_load_path, model_save_path=model_save_path)
    elif choice == "3":
        model_load_path = input("Enter model load path (e.g., models/ai_model.h5): ")
        start_game_with_ai(model_load_path=model_load_path)
    else:
        print("Invalid choice!")