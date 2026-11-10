import numpy as np
import pickle
import os

class Agent_Q_learning():
    def __init__(self,
                env,
                learning_rate,
                start_epsilon,
                epsilon_decay,
                final_epsilon,
                gamma,
                table_directory=None
                ):
        self.env=env
        self.learning_rate=learning_rate
        self.epsilon=start_epsilon
        self.epsilon_decay=epsilon_decay
        self.final_epsilon=final_epsilon
        self.gamma=gamma

        self.basic_action = self.env.action_space.sample()

        if (table_directory):
            with open(table_directory, "rb") as file:
                self.Q_table = pickle.load(file)
        else :
            self.Q_table={}


    def best_action(self,state):
        if (state in self.Q_table.keys()):
            actions_stored=self.Q_table[state]
            action= max(actions_stored, key=lambda key: actions_stored[key])
        else :
            action=self.basic_action

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


    def q_value_max(self,state):
        if (state in self.Q_table.keys()):
            q_value=max(self.Q_table[state].values())
        else :
            q_value=0
        return q_value
    

    def update(self,
               state,
               action,
               next_state, 
               reward,
               terminated
               ):

        if (state in self.Q_table.keys()):
            if (action in self.Q_table[state].keys()):
                current_value=self.Q_table[state][action]
            else :
                current_value=0
        else:
            current_value=0

        error_TD=reward + ((self.gamma*self.q_value_max(next_state))*(1 - terminated))-current_value

        if (not (state in self.Q_table.keys())):
            self.Q_table[state]={}


        self.Q_table[state][action]=current_value + (self.learning_rate*error_TD)

        return error_TD

    def save_table(self, training_name, directory="models"):
        os.makedirs(directory, exist_ok=True)

        path = os.path.join(directory, f"{training_name}.pkl")

        with open(path, "wb") as file:
            pickle.dump(self.Q_table, file)

        print(f"Q-table saved: {path}")
