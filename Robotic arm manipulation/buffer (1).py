#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


#(state,action,reward,next_state,done)


# In[ ]:


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size  
        self.mem_cntr = 0  # Use consistent naming - includes the 'n'
        
        # Initialize memory buffers
        self.state_memory = np.zeros((max_size, *input_shape))
        self.next_state_memory = np.zeros((max_size, *input_shape))
        self.action_memory = np.zeros((max_size, n_actions))
        self.reward_memory = np.zeros(max_size)
        self.terminal_memory = np.zeros(max_size, dtype=np.bool_)
        
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size  # Uses mem_cntr
        
        self.state_memory[index] = state
        self.next_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        # Make sure to use mem_cntr here too
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, next_states, dones
   

