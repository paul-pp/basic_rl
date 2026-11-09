import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from Q_learning import *
import matplotlib.pyplot as plt


# Create a simple environment perfect for beginners
env_lake = gym.make('FrozenLake-v1',map_name="4x4")
# env_lake = gym.make('LunarLander-v3')


n_episodes=50000
start_eps=1
agent_lake=Agent(
    env=env_lake,
    learning_rate=1e-2,
    start_epsilon=start_eps,
    epsilon_decay = start_eps / (n_episodes/2),
    final_epsilon=0.1,
    gamma=0.99
)





# print(f"Starting observation: {observation}")

display=1000
Liste_reward=[]
Liste_mean_reward_per_display=[]

for episode in range(n_episodes):
    # Reset environment to start a new episode
    current_observation, info = env_lake.reset()

    episode_over = False
    total_reward = 0

    success=0
    while not episode_over:
        action=agent_lake.eps_greedy_action(state=current_observation)

        # Take the action and see what happens
        new_observation, reward, terminated, truncated, info = env_lake.step(action)

        agent_lake.update(
                        state=current_observation,
                       action=action,
                       next_state=new_observation, 
                       reward=reward,
                       terminated=terminated
                       )


        current_observation=new_observation

        
        if (reward==1):
            success=1
        
        total_reward += reward
        episode_over = terminated or truncated

    agent_lake.update_epsilon()

    # Liste_reward.append(total_reward)
    Liste_reward.append(success)

    if (episode%display==0):
        value=sum(Liste_reward[episode-display:episode])/display
        print(f"episode {episode}:",value)
        Liste_mean_reward_per_display.append(value)


# plt.plot(Liste_reward)
plt.plot(display*np.arange(len(Liste_mean_reward_per_display)),Liste_mean_reward_per_display)

# plt.xscale("log")  
plt.show()
env_lake.close()


print("\ntest phase")

env_lake = gym.make('FrozenLake-v1',map_name="4x4",render_mode='human')

Liste_reward=[]
##### TEST #####
for episode in range(10):
    current_observation, info = env_lake.reset()

    episode_over = False
    total_reward = 0

    success=0
    while not episode_over:
        action=agent_lake.best_action(state=current_observation)

        new_observation, reward, terminated, truncated, info = env_lake.step(action)

        current_observation=new_observation

        total_reward += reward
        episode_over = terminated or truncated

        if (reward==1):
            success=1

    # Liste_reward.append(total_reward)
    Liste_reward.append(success)

    if (episode%display==0):

        print(sum(Liste_reward[episode-display:episode])/display)


env_lake.close()

