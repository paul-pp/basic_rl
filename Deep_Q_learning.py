import numpy as np
import torch
import torch.nn as nn
import copy
import os
from collections import deque
import random



class Agent_DQN():
    def __init__(self,
                env,

                start_learning_rate,
                start_factor_lr,
                end_factor_lr,
                total_iters_lr,

                start_epsilon,
                epsilon_decay,
                final_epsilon,

                gamma,

                main_neural_network_directory=None
                ):

        
        self.env=env

        self.epsilon=start_epsilon
        self.epsilon_decay=epsilon_decay
        self.final_epsilon=final_epsilon

        self.gamma=gamma

        self.make_nn_structure()

        if (main_neural_network_directory):
            state_dict = torch.load(main_neural_network_directory)
            self.main_network.load_state_dict(state_dict)

        self.criterion = torch.nn.MSELoss(reduction='mean')
        if (start_learning_rate):
            self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=start_learning_rate)
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=start_factor_lr,
                        end_factor=end_factor_lr,
                        total_iters=total_iters_lr
                    )

            # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            #     self.optimizer,
            #     gamma=0.99998
            # )
        

        self.network_copy=copy.deepcopy(self.main_network)

    


    def make_nn_structure(self):
        n_in=self.env.observation_space.shape[0]
        n_h1=100
        n_h2=100
        n_out=self.env.action_space.n

        self.main_network = nn.Sequential(
            nn.Linear(n_in, n_h1),
            nn.ReLU(),

            nn.Linear(n_h1, n_h2),
            nn.ReLU(),

            nn.Linear(n_h2, n_out)
        )

        return 0




    def normalize_state(self,state):
        min_state=np.array(self.env.observation_space.low)
        max_state=np.array(self.env.observation_space.high)
        state=np.array(state)

        # norm_state=(state-min_state)/(max_state-min_state)
        # return norm_state
        return state


    def best_action(self,state):

        norm_state=self.normalize_state(state)
        state_tensor=torch.tensor(norm_state)
        
        self.main_network.eval()
        prediction=self.main_network(state_tensor)
        action=int(torch.argmax(prediction,dim=-1))
        return action


    def update_epsilon(self):
        self.epsilon=max(self.final_epsilon,self.epsilon-self.epsilon_decay)
        return self.epsilon

    
    def eps_greedy_action(self,state):
        rv=np.random.uniform()
        if (rv<self.epsilon):
            action = self.env.action_space.sample()
        else :
            action=self.best_action(state)
        return action


    def make_target(self,Liste_rewards_tensor,Liste_next_states,Liste_terminated_tensor):
        self.network_copy.eval()
        predictions=self.network_copy(Liste_next_states)

        target_tensor=Liste_rewards_tensor + ((self.gamma*torch.max(predictions,dim=-1).values)*(1-Liste_terminated_tensor))
        
        return target_tensor



    def update(self,
               Liste_states,
                Liste_actions,
                Liste_rewards,
                Liste_next_states,
                Liste_terminated
               ):

        Liste_states_norm=self.normalize_state(Liste_states)
        Liste_next_states_norm=self.normalize_state(Liste_next_states)
                        
        Liste_states_tensor=torch.tensor(Liste_states_norm,dtype=torch.float32)
        Liste_actions_tensor=torch.tensor(Liste_actions)
        Liste_rewards_tensor=torch.tensor(Liste_rewards,dtype=torch.float32)
        Liste_next_states_tensor=torch.tensor(Liste_next_states_norm,dtype=torch.float32)
        Liste_terminated_tensor=torch.tensor(Liste_terminated,dtype=torch.float32)

        y=self.make_target(Liste_rewards_tensor,Liste_next_states_tensor,Liste_terminated_tensor)


        self.main_network.train()
            
        predictions=self.main_network(Liste_states_tensor)
        q_values=predictions.gather(
            1,
            Liste_actions_tensor.unsqueeze(1)
        ).squeeze(1)

        loss = self.criterion(q_values, y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def save_network(self,training_name,directory="models"):
        os.makedirs(directory, exist_ok=True)

        path = os.path.join(directory, f"{training_name}.pth")

        torch.save(
            self.main_network.state_dict(),
            path
        )

        print(f"Model saved: {path}")
        




class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):
        return len(self.buffer)
