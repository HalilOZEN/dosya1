import numpy as np
import random
import matplotlib.pyplot as plt

# Labirent boyutları
n_rows, n_cols = 6, 6

# Ödül ve ceza değerleri
reward_exit = 100
penalty_step = -1

# Labirent tanımı (0 = boş, 1 = duvar, 2 = çıkış)
maze = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 2]
])

# Q-Tablosu ve parametreler
q_table = np.zeros((n_rows, n_cols, 4)) # 4 hareket yönü: 0=üst, 1=sağ, 2=alt, 3=sol
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005
episodes = 1000
max_steps = 100

# Yardımcı fonksiyonlar
def is_valid_move(maze, position):
    x, y = position
    return 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and maze[x, y] != 1

def get_next_position(position, action):
    x, y = position
    if action == 0: return (x-1, y) # Üst
    if action == 1: return (x, y+1) # Sağ
    if action == 2: return (x+1, y) # Alt
    if action == 3: return (x, y-1) # Sol

def get_reward(maze, position):
    if maze[position] == 2:
        return reward_exit
    return penalty_step

# Q-öğrenme algoritması
for episode in range(episodes):
    state = (0, 0)
    for step in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1, 2, 3])
        else:
            action = np.argmax(q_table[state[0], state[1], :])

        next_state = get_next_position(state, action)
        
        if not is_valid_move(maze, next_state):
            next_state = state

        reward = get_reward(maze, next_state)

        q_value = q_table[state[0], state[1], action]
        best_next_q = np.max(q_table[next_state[0], next_state[1], :])

        q_table[state[0], state[1], action] = q_value + learning_rate * (reward + discount_factor * best_next_q - q_value)
        
        state = next_state

        if maze[state] == 2:
            break

    epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate * episode))

# Öğrenilen en iyi yolu gösterme
state = (0, 0)
steps = [state]

while maze[state] != 2:
    action = np.argmax(q_table[state[0], state[1], :])
    state = get_next_position(state, action)
    steps.append(state)

print("En kısa yol:", steps)

# Labirent ve yolun görselleştirilmesi
for step in steps:
    maze[step] = 3

plt.imshow(maze, cmap='hot')
plt.colorbar()
plt.show()
