import gymnasium as gym
from Deep_Q_learning import *
import matplotlib.pyplot as plt


env = gym.make('LunarLander-v3')
# env = gym.make('LunarLander-v3',render_mode='human')


n_episodes=500
start_eps=1.0
batch_size=64
update_weights_period=1000


display_train=100
display_test=1000

agent=Agent(
    env=env,

    start_learning_rate=0.002,
    start_factor_lr=1.0,
    end_factor_lr=1.0,
    total_iters_lr=int(n_episodes/2),
    
    start_epsilon=start_eps,
    epsilon_decay = start_eps / (n_episodes/2),
    final_epsilon=0.1,
    gamma=0.99
    )
buffer = ReplayBuffer(capacity=100_000)



# Définition d'un Interactive mode
plt.ion() 
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot([], [], label="Mean reward")
ax.set_xlabel("Episode")
ax.set_ylabel("Mean reward")
# ax.set_title("DQN - LunarLander training")
ax.legend()
ax.grid()
fig.show()



Liste_reward_train=[]
Liste_mean_reward_per_display_train=[]


Liste_states=[]
Liste_actions=[]
Liste_rewards=[]
Liste_terminated=[]

count_update=0
episode_lenght=0
for episode in range(n_episodes):
    # Reset environment to start a new episode
    current_observation, info = env.reset()

    episode_over = False
    total_reward = 0

    success=0
    while not episode_over:
        episode_lenght+=1

        action=agent.eps_greedy_action(state=current_observation)
        new_observation, reward, terminated, truncated, info = env.step(action)
        buffer.add(current_observation, action, reward, new_observation, int(terminated))



        current_observation=new_observation
        episode_over = terminated or truncated


        if (len(buffer)>batch_size):
            # print("len(Liste_states)",len(Liste_states))
            count_update+=1
            Liste_states,Liste_actions,Liste_rewards,Liste_next_states,Liste_terminated=buffer.sample(batch_size)
            agent.update(
                        Liste_states=Liste_states,
                        Liste_actions=Liste_actions,
                        Liste_rewards=Liste_rewards,
                        Liste_next_states=Liste_next_states,
                        Liste_terminated=Liste_terminated
                        )

            Liste_states=[]
            Liste_actions=[]
            Liste_rewards=[]
            Liste_terminated=[]

        if (count_update>update_weights_period):
            count_update=0
            agent.network_copy.load_state_dict(agent.main_network.state_dict())


        total_reward += reward

    agent.update_epsilon()
    agent.scheduler.step()

    Liste_reward_train.append(total_reward)




    if (episode!=0 and episode%display_train==0):
        value=sum(Liste_reward_train[episode-display_train:episode])/display_train
        # print(f"episode {episode}:",value)
        current_lr = agent.optimizer.param_groups[0]['lr']
        print(f"\nEpisode {episode} | LR = {current_lr:.6f}")
        print(f"Episode mean lenght:",episode_lenght/display_train)
        episode_lenght=0
        Liste_mean_reward_per_display_train.append(value)

        x = display_train * np.arange(
        len(Liste_mean_reward_per_display_train)
        )

        y = Liste_mean_reward_per_display_train

        line.set_data(x, y)

        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        plt.pause(0.01)


agent.save_network("model_luner_lander")
print("Apprentissage terminé !")

plt.ioff()
plt.show()
env.close()







# ##### TEST #####
env = gym.make('LunarLander-v3',render_mode='human')

for episode in range(10):
    # Reset environment to start a new episode
    current_observation, info = env.reset()

    episode_over = False
    total_reward = 0

    success=0
    while not episode_over:
        action=agent.best_action(state=current_observation)

        # Take the action and see what happens
        new_observation, reward, terminated, truncated, info = env.step(action)

        current_observation=new_observation

        total_reward += reward
        episode_over = terminated or truncated


env.close()

