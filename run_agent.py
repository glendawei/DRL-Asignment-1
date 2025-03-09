import importlib.util
import time
import numpy as np
import student_agent  # 改為 student_agent
from simple_custom_taxi_env import SimpleTaxiEnv

def run_agent(render=False, train=True, episodes=5000):
    env = SimpleTaxiEnv(fuel_limit=5000)

    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            step_count = 0

            while not done:
                action = student_agent.get_action(obs)  # 取得動作
                next_obs, reward, done, _, _ = env.step(action)


                if train:
                    student_agent.update_Q_table(obs, action, reward, next_obs)

                obs = next_obs
                total_reward += reward
                step_count += 1

                if render and episode % 10 == 0:  # 只顯示每 10 回合一次，避免影響速度
                    taxi_row, taxi_col, *_ = obs
                    env.render_env(taxi_pos=(taxi_row, taxi_col), action=action, step=step_count, fuel=env.current_fuel)
                    time.sleep(0.2)

            if train:
                student_agent.exploration_rate = max(student_agent.min_exploration, student_agent.exploration_rate * student_agent.exploration_decay)

            print(f"Episode {episode + 1}: Steps = {step_count}, Total Reward = {total_reward}")

            # **每 1000 回合存一次 Q-table**
            if train and (episode + 1) % 1000 == 0:
                student_agent.save_Q_table()
                print("✅ Q-table saved at episode", episode + 1)

    except KeyboardInterrupt:
        print("\n⏹ 訓練被手動中止！正在儲存 Q-table...")
        student_agent.save_Q_table()
        print("✅ Q-table 已儲存，可以下次繼續訓練。")

if __name__ == "__main__":
    run_agent(render=True, train=True, episodes=5000)
