import numpy as np
import pickle
import random
import os

# Q-learning 參數
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1.0
exploration_decay = 0.995
min_exploration = 0.01

# 嘗試載入 Q-table
Q_table = {}
q_table_file = "q_table.pkl"

if os.path.exists(q_table_file):
    with open(q_table_file, "rb") as f:
        Q_table = pickle.load(f)
    print("Q-table loaded successfully!")
else:
    print("No Q-table found, starting fresh.")

def get_action(obs):
    """ 選擇最佳動作（帶有 epsilon-greedy 探索策略） """
    global exploration_rate

    state = tuple(obs)  # 轉換 state 格式，確保可以存入字典

    if random.uniform(0, 1) < exploration_rate or state not in Q_table:
        return random.choice([0, 1, 2, 3, 4, 5])  # 探索：隨機選擇動作

    # 利用 Q-table 來選擇最好的行動
    return max(Q_table[state], key=Q_table[state].get)

def update_Q_table(state, action, reward, next_state):
    """ 使用 Q-learning 更新 Q-table """
    global Q_table

    state, next_state = tuple(state), tuple(next_state)

    if state not in Q_table:
        Q_table[state] = {a: 0 for a in range(6)}

    if next_state not in Q_table:
        Q_table[next_state] = {a: 0 for a in range(6)}

    best_next_action = max(Q_table[next_state], key=Q_table[next_state].get)

    Q_table[state][action] = (1 - learning_rate) * Q_table[state][action] + \
        learning_rate * (reward + discount_factor * Q_table[next_state][best_next_action])

def save_Q_table():
    """ 儲存 Q-table 以便下次繼續訓練 """
    with open(q_table_file, "wb") as f:
        pickle.dump(Q_table, f)
    print("Q-table saved successfully!")
