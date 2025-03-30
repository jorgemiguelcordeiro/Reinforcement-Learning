import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import *
from networks import *

class Agent:
    
    def __init__(self, actor_learning_rate, critic_learning_rate, input_dims, tau, env, gamma=0.99, update_actor_interval=2, warmup=100, n_actions=2, max_size=1000000, layer1_size=256, layer2_size=128, batch_size=100, noise=0.1):
        
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0  # Initialize time step counter
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        
        # Create the networks
        self.actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                 fc2_dims=layer2_size, n_actions=n_actions,
                                 name='actor', learning_rate=actor_learning_rate)
        self.critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                     fc2_dims=layer2_size, n_actions=n_actions,
                                     name='critic_1', learning_rate=critic_learning_rate)
        self.critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                     fc2_dims=layer2_size, n_actions=n_actions,
                                     name='critic_2', learning_rate=critic_learning_rate)

        #create the target networks
        self.target_actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                 fc2_dims=layer2_size, n_actions=n_actions,
                                 name='target_actor', learning_rate=actor_learning_rate)

        self.target_critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                     fc2_dims=layer2_size, n_actions=n_actions,
                                     name='target_critic_1', learning_rate=critic_learning_rate)
        self.target_critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                     fc2_dims=layer2_size, n_actions=n_actions,
                                     name='target_critic_2', learning_rate=critic_learning_rate)
        
        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, validation=False):
        # Increment time step counter
        self.time_step += 1
        
        if self.time_step < self.warmup:
            action = np.random.uniform(
                low=self.min_action, 
                high=self.max_action, 
                size=self.n_actions
            )
            #print(f"Warmup action: {action}")
            return action
        
        # Handle potentially problematic observation formats
        if isinstance(observation, (float, int, np.float64, np.int64)) or (
                isinstance(observation, np.ndarray) and (len(observation.shape) == 0 or observation.size == 1)):
            observation = np.zeros(self.actor.input_dims)
            print(f"WARNING: Reconstructed observation: {observation}")
        
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        if state.dim() == 1:  # Add batch dimension if needed
            state = state.unsqueeze(0)
        
        # Get action from actor network
        mu = self.actor.forward(state)
        
        # Add noise for exploration (if not in validation mode)
        if not validation:
            noise = T.tensor(np.random.normal(scale=self.noise, size=(1, self.n_actions)), 
                            dtype=T.float).to(self.actor.device)
            mu_prime = mu + noise
        else:
            mu_prime = mu
        
        # Clamp to valid action range
        mu_prime = T.clamp(mu_prime, 
                        T.tensor(self.min_action, dtype=T.float).to(self.actor.device), 
                        T.tensor(self.max_action, dtype=T.float).to(self.actor.device))
        
        # Return the action as numpy array
        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size * 10:
            return
            
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
                
        # Make sure dimensions are correct
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)

        target_actions = self.target_actor.forward(next_state)
        # Add noise to target actions for smoothing
        noise = T.clamp(
            T.tensor(np.random.normal(scale=0.2, size=target_actions.shape), dtype=T.float).to(self.critic_1.device),
            -0.5, 0.5
        )
        target_actions = target_actions + noise
        target_actions = T.clamp(target_actions, 
                               T.tensor(self.min_action[0], dtype=T.float).to(self.critic_1.device),
                               T.tensor(self.max_action[0], dtype=T.float).to(self.critic_1.device))
        
        next_q1 = self.target_critic_1.forward(next_state, target_actions)
        next_q2 = self.target_critic_2.forward(next_state, target_actions)
        
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)
        
        next_q1[done] = 0.0
        next_q2[done] = 0.0
        
        next_q1 = next_q1.view(-1)
        next_q2 = next_q2.view(-1)

        next_critic_value = T.min(next_q1, next_q2)

        target = reward + self.gamma*next_critic_value
        target = target.view(self.batch_size, 1)
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        self.learn_step_cntr += 1
        
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return
        
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()

        self.actor.optimizer.step()
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
            
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()
        
        actor_state_dict = dict(actor_params)
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)
        
        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + (1-tau)*target_critic_1_state_dict[name].clone()
        
        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + (1-tau)*target_critic_2_state_dict[name].clone()
        
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()
        
        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        print("Saving checkpoint...")
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        try:
            print("Loading checkpoint...")
            self.actor.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            self.target_critic_1.load_checkpoint()
            self.target_critic_2.load_checkpoint()
            print("Successfully loaded models")
        except:
            print("Failed to load models. Starting from scratch")
