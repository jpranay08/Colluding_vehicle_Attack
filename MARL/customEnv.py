
import sys
sys.path.append('.')
import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        # Define the smoothing factors for reputation update policy
       
        
        # Define the state variables
        self.previous_smoothing_factor = None
        self.average_reputation_score = None
        self.average_true_feedbacks = None
        
        # Set the ranges and data types for the state variables
        self.previous_smoothing_factor_range = (0.0, 1.0)
        self.average_reputation_score_range = (0.0, 1.0)
        self.average_true_feedbacks_range = (0.0, 1.0)
        self.previous_smoothing_factor_dtype = np.float32
        self.average_reputation_score_dtype = np.float32
        self.average_true_feedbacks_dtype = np.float32
        
        # Create the observation space with appropriate ranges and data types
        self.observation_space = spaces.Box(
            low=np.array([self.previous_smoothing_factor_range[0], self.average_reputation_score_range[0], self.average_true_feedbacks_range[0]]),
            high=np.array([self.previous_smoothing_factor_range[1], self.average_reputation_score_range[1], self.average_true_feedbacks_range[1]]),
            dtype=np.float32
        )
        
        # Create the action space for selecting the smoothing factor
        self.action_space = spaces.Box(low=np.array([self.previous_smoothing_factor_range[0]]),high=np.array([self.previous_smoothing_factor_range[1]]),dtype=np.float32)
        self.episode_length=100
        self.step_num=0
        
    def dqreset(self):
        # Reset the environment and return the initial state
        self.previous_smoothing_factor = np.random.uniform(self.previous_smoothing_factor_range[0], self.previous_smoothing_factor_range[1])
        self.average_reputation_score = np.random.uniform(self.average_reputation_score_range[0], self.average_reputation_score_range[1])
        self.average_true_feedbacks = np.random.randint(self.average_true_feedbacks_range[0], self.average_true_feedbacks_range[1])
        self.step_num=0
        return self._get_state()
    
    def step(self, action, currreward, average_reputation_score):
        # Update the state based on the selected action and return the new state, reward, and done flag
        #print("inside the rl env action is \n\n",action,"\n \n \n ")
        smoothing_factor = action
        
        
        self.previous_smoothing_factor = smoothing_factor
        self.average_reputation_score = average_reputation_score
        self.average_true_feedbacks = currreward
        
        state = self._get_state()
        reward = self.average_true_feedbacks
        self.step_num += 1
        done = False  
        if self.step_num >=self.episode_length:
            done=True
        
        return state, reward, done, {}
    
    def _get_state(self):
        # Return the current state as a numpy array
        state = np.array([self.previous_smoothing_factor, self.average_reputation_score, self.average_true_feedbacks])
        state[0] = round(state[0], 2)  # Limit the first variable to two decimal points
        return state
        #return np.array([self.previous_smoothing_factor, self.average_reputation_score, self.average_true_feedbacks])
    
    
    # def _calculate_reward(self):
        # Implement your reward calculation logic
    # return 0.0
