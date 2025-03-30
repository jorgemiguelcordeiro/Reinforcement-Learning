# TD3 Reinforcement Learning for Robotic Door Opening

This repository implements a Twin Delayed Deep Deterministic Policy Gradient (TD3) reinforcement learning agent for a robotic door opening task using the RoboSuite environment with a Panda robot arm.

![Robotic Door Opening Task](https://robosuite.ai/images/tasks/door.png)

## Project Overview

This project uses the TD3 algorithm to train a robotic arm to open a door. The TD3 algorithm improves upon DDPG by addressing function approximation errors through:

1. Twin critics to reduce overestimation bias
2. Delayed policy updates for more stable learning
3. Target policy smoothing to prevent exploitation of Q-function errors

The agent learns to interact with a door environment, needing to approach, grasp, and turn the handle to open the door.

## Code Structure

- `main.ipynb`: Main script for training and testing the agent
- `td3_torch.py`: TD3 agent implementation
- `networks.py`: Neural network architectures for actor and critics
- `buffer.py`: Replay buffer implementation for experience storage


## Architecture

### TD3 Agent
The TD3 agent consists of:
- An actor network that determines actions based on states
- Twin critic networks that estimate Q-values
- Target networks for stable learning
- Experience replay buffer for off-policy learning

### Hyperparameters
- Actor learning rate: 0.0003
- Critic learning rate: 0.0003
- Discount factor (gamma): 0.99
- Replay buffer size: 1,000,000
- Batch size: 256
- Target network update rate (tau): 0.005
- Exploration noise: 0.1
- Actor network: [400, 300] hidden units
- Critic network: [400, 300] hidden units

## Features

- **State-of-the-art TD3 algorithm**: Handles continuous action spaces and mitigates overestimation bias
- **Robotic manipulation in simulation**: Uses RoboSuite (built on MuJoCo) for realistic physics 
- **TensorBoard integration**: Visualize learning curves and performance metrics
- **Checkpoint saving/loading**: Resume training from previous checkpoints
- **Customizable hyperparameters**: Easily modify learning parameters

## Results

## Future Improvements

Several avenues for improvement could enhance the agent's performance:

### Algorithm Enhancements
- **Prioritized Experience Replay**: Weight experiences by their TD error to learn more efficiently from important transitions
- **Distributional RL**: Model the full distribution of returns rather than just the mean
- **Hierarchical RL**: Implement a hierarchical structure for handling subtasks (approach, grasp, turn)
- **Ensemble methods**: Use multiple actors to improve exploration and stability

### Training Optimizations
- **Curriculum Learning**: Gradually increase task difficulty by starting with the arm closer to the door
- **Domain Randomization**: Vary physical parameters to improve robustness
- **Hyperparameter tuning**: Systematically explore learning rates, network sizes, and other parameters
- **Exploration strategies**: Implement parameter space noise or curiosity-driven exploration

### Environment Modifications
- **Dense reward shaping**: Design more informative reward functions
- **Multi-camera observations**: Add additional viewpoints to enhance state representation
- **Demonstration data**: Incorporate human demonstrations to bootstrap learning
- **Multi-task learning**: Train on multiple door configurations simultaneously

### Real-World Transfer
- **Sim-to-real transfer**: Add techniques to help transfer learned policies to physical robots
- **Domain adaptation**: Reduce the reality gap through calibration and adaptation techniques
- **Robust policy optimization**: Optimize for worst-case scenarios to handle real-world uncertainty

### Visualization and Analysis
- **Attention mechanisms**: Visualize which state components are most important for the agent
- **Interpretability tools**: Develop methods to understand the agent's decision-making process
- **Benchmark comparisons**: Compare against other algorithms (SAC, PPO, etc.)

