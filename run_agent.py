import threading
import time
import student_agent
from simple_custom_taxi_env import SimpleTaxiEnv

# 定義單個thread運行的function
def run_single_episode(episode, render=False, train=True):
    env = SimpleTaxiEnv(fuel_limit=5000)
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action = student_agent.get_action(obs)
        next_obs, reward, done, _, _ = env.step(action)

        if train:
            student_agent.update_Q_table(obs, action, reward, next_obs)

        obs = next_obs
        total_reward += reward
        step_count += 1

        if render and episode % 10 == 0:
            taxi_row, taxi_col, *_ = obs
            env.render_env(taxi_pos=(taxi_row, taxi_col), action=action, step=step_count, fuel=env.current_fuel)
            time.sleep(0.2)

    if train:
        student_agent.exploration_rate = max(student_agent.min_exploration,
                                             student_agent.exploration_rate * student_agent.exploration_decay)

    print(f"Episode {episode + 1}: Steps = {step_count}, Total Reward = {total_reward}")

# 多執行緒版本
def run_agent_multithread(render=False, train=True, episodes=5000, num_threads=5):
    threads = []

    try:
        for episode in range(episodes):
            t = threading.Thread(target=run_single_episode, args=(episode, render, train))
            threads.append(t)
            t.start()

            # 若達到最大執行緒數則等待執行緒結束後再繼續
            if len(threads) >= num_threads:
                for th in threads:
                    th.join()
                threads = []

            if train and (episode + 1) % 1000 == 0:
                student_agent.save_Q_table()
                print("✅ Q-table saved at episode", episode + 1)

        # 確保所有剩餘執行緒都完成
        for th in threads:
            th.join()

    except KeyboardInterrupt:
        print("\n⏹ 訓練被手動中止！正在儲存 Q-table...")
        student_agent.save_Q_table()
        print("✅ Q-table 已儲存，可以下次繼續訓練。")

if __name__ == "__main__":
    run_agent_multithread(render=True, train=True, episodes=5000, num_threads=10)