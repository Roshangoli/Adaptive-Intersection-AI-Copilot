#!/usr/bin/env python3
"""
Reinforcement Learning Agent for Adaptive Intersection AI Copilot.
This module implements PPO and DQN agents for traffic light control optimization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficEnvironment:
    """Environment wrapper for traffic control."""
    
    def __init__(self, state_dim: int = 8, action_dim: int = 4):
        """
        Initialize traffic environment.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Traffic light phases
        self.phases = [
            "GGrrrrGGrrrr",  # North-South green
            "yyyrrryyyrrr",  # North-South yellow
            "rrrGGGrrrGGG",  # East-West green
            "rrryyyrrryyy"   # East-West yellow
        ]
        
        # Environment parameters
        self.min_phase_time = 5  # Minimum phase duration in seconds
        self.max_phase_time = 60  # Maximum phase duration in seconds
        
    def get_state(self, traffic_data: Dict) -> np.ndarray:
        """
        Extract state from traffic data.
        
        Args:
            traffic_data: Dictionary containing traffic information
            
        Returns:
            State vector
        """
        # Normalize state components
        state = np.zeros(self.state_dim)
        
        # Vehicle counts per direction (normalized)
        state[0] = min(traffic_data.get('vehicles_ns', 0) / 10.0, 1.0)
        state[1] = min(traffic_data.get('vehicles_ew', 0) / 10.0, 1.0)
        
        # Pedestrian counts per direction (normalized)
        state[2] = min(traffic_data.get('pedestrians_ns', 0) / 20.0, 1.0)
        state[3] = min(traffic_data.get('pedestrians_ew', 0) / 20.0, 1.0)
        
        # Waiting times (normalized)
        state[4] = min(traffic_data.get('avg_vehicle_wait', 0) / 30.0, 1.0)
        state[5] = min(traffic_data.get('avg_pedestrian_wait', 0) / 60.0, 1.0)
        
        # Time features
        state[6] = traffic_data.get('hour', 12) / 24.0  # Hour of day
        state[7] = traffic_data.get('is_weekend', 0)  # Weekend flag
        
        return state
    
    def calculate_reward(self, state: np.ndarray, action: int, 
                        next_state: np.ndarray) -> float:
        """
        Calculate reward based on state, action, and next state.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Penalty for waiting vehicles (higher weight)
        reward -= state[4] * 10.0  # Vehicle wait time
        
        # Penalty for waiting pedestrians (highest weight)
        reward -= state[5] * 20.0  # Pedestrian wait time
        
        # Bonus for throughput
        reward += (state[0] + state[1]) * 2.0  # Vehicle counts
        reward += (state[2] + state[3]) * 3.0  # Pedestrian counts
        
        # Penalty for frequent phase changes (encourage stability)
        if action != 0:  # Assuming 0 is "no change"
            reward -= 1.0
        
        # Bonus for reducing wait times
        if next_state[4] < state[4]:  # Vehicle wait time improved
            reward += 5.0
        if next_state[5] < state[5]:  # Pedestrian wait time improved
            reward += 10.0
        
        return reward

class DQNAgent:
    """Deep Q-Network agent for traffic control."""
    
    def __init__(self, state_dim: int = 8, action_dim: int = 4, 
                 learning_rate: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 10000):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            memory_size: Size of replay buffer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural network
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training parameters
        self.batch_size = 32
        self.update_target_freq = 100
        self.step_count = 0
        
    def _build_network(self) -> nn.Module:
        """Build the Q-network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Action to take
        """
        if training and np.random.random() <= self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self) -> float:
        """
        Train the agent on a batch of experiences.
        
        Returns:
            Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
        logger.info(f"DQN model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        logger.info(f"DQN model loaded from {filepath}")

class PPOAgent:
    """Proximal Policy Optimization agent for traffic control."""
    
    def __init__(self, state_dim: int = 8, action_dim: int = 4,
                 learning_rate: float = 0.0003, gamma: float = 0.99,
                 clip_ratio: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            clip_ratio: PPO clip ratio
            value_coef: Value function loss coefficient
            entropy_coef: Entropy loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Neural networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )
        
        # Training parameters
        self.batch_size = 64
        self.epochs = 4
        
    def _build_actor(self) -> nn.Module:
        """Build the actor network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Softmax(dim=-1)
        )
    
    def _build_critic(self) -> nn.Module:
        """Build the critic network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def act(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Choose action using current policy.
        
        Args:
            state: Current state
            
        Returns:
            Action, log probability, value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action probabilities
        action_probs = self.actor(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Get value
        value = self.critic(state_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate states and actions.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            Log probabilities, values, entropy
        """
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states)
        
        return log_probs, values, entropy
    
    def update(self, states: np.ndarray, actions: np.ndarray, 
               rewards: np.ndarray, dones: np.ndarray, 
               old_log_probs: np.ndarray, values: np.ndarray):
        """
        Update the agent using PPO.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            dones: Batch of done flags
            old_log_probs: Batch of old log probabilities
            values: Batch of old values
        """
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        old_values = torch.FloatTensor(values)
        
        # Calculate advantages
        advantages = self._calculate_advantages(rewards, dones, old_values)
        returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        for _ in range(self.epochs):
            # Get current policy
            log_probs, new_values, entropy = self.evaluate(states, actions)
            
            # Calculate ratios
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Calculate surrogates
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            # Calculate losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            entropy_loss = -entropy.mean()
            
            total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()
    
    def _calculate_advantages(self, rewards: np.ndarray, dones: np.ndarray, 
                            values: torch.Tensor) -> torch.Tensor:
        """Calculate advantages using GAE."""
        advantages = []
        advantage = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                advantage = 0
            advantage = rewards[i] + self.gamma * advantage - values[i].item()
            advantages.insert(0, advantage)
        
        return torch.FloatTensor(advantages)
    
    def save(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
        logger.info(f"PPO model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"PPO model loaded from {filepath}")

class TrafficController:
    """Main traffic controller using RL agent."""
    
    def __init__(self, agent_type: str = "dqn", **kwargs):
        """
        Initialize traffic controller.
        
        Args:
            agent_type: Type of RL agent ("dqn" or "ppo")
            **kwargs: Additional arguments for agent
        """
        self.environment = TrafficEnvironment()
        
        if agent_type == "dqn":
            self.agent = DQNAgent(**kwargs)
        elif agent_type == "ppo":
            self.agent = PPOAgent(**kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.agent_type = agent_type
        self.current_phase = 0
        self.phase_start_time = 0
        
    def get_action(self, traffic_data: Dict) -> int:
        """
        Get action from RL agent.
        
        Args:
            traffic_data: Current traffic data
            
        Returns:
            Action (phase change)
        """
        state = self.environment.get_state(traffic_data)
        
        if self.agent_type == "dqn":
            action = self.agent.act(state, training=False)
        else:  # PPO
            action, _, _ = self.agent.act(state)
        
        return action
    
    def update_phase(self, action: int, current_time: int) -> Dict:
        """
        Update traffic light phase based on action.
        
        Args:
            action: Action from RL agent
            current_time: Current simulation time
            
        Returns:
            Phase update information
        """
        # Check if minimum phase time has elapsed
        if current_time - self.phase_start_time >= self.environment.min_phase_time * 10:
            self.current_phase = action
            self.phase_start_time = current_time
        
        return {
            'phase': self.current_phase,
            'phase_duration': current_time - self.phase_start_time,
            'phase_state': self.environment.phases[self.current_phase]
        }
    
    def train(self, experiences: List[Dict]) -> float:
        """
        Train the RL agent.
        
        Args:
            experiences: List of experience dictionaries
            
        Returns:
            Training loss
        """
        if not experiences:
            return 0.0
        
        if self.agent_type == "dqn":
            # Store experiences in replay buffer
            for exp in experiences:
                self.agent.remember(
                    exp['state'], exp['action'], exp['reward'],
                    exp['next_state'], exp['done']
                )
            
            # Train on batch
            return self.agent.replay()
        
        else:  # PPO
            # Extract data
            states = np.array([exp['state'] for exp in experiences])
            actions = np.array([exp['action'] for exp in experiences])
            rewards = np.array([exp['reward'] for exp in experiences])
            dones = np.array([exp['done'] for exp in experiences])
            old_log_probs = np.array([exp['log_prob'] for exp in experiences])
            values = np.array([exp['value'] for exp in experiences])
            
            # Update agent
            self.agent.update(states, actions, rewards, dones, old_log_probs, values)
            return 0.0  # PPO doesn't return loss in this implementation
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        self.agent.save(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.agent.load(filepath)

# Example usage
if __name__ == "__main__":
    # Initialize controller
    controller = TrafficController(agent_type="dqn")
    
    # Sample traffic data
    traffic_data = {
        'vehicles_ns': 5,
        'vehicles_ew': 3,
        'pedestrians_ns': 8,
        'pedestrians_ew': 4,
        'avg_vehicle_wait': 10.0,
        'avg_pedestrian_wait': 15.0,
        'hour': 14,
        'is_weekend': 0
    }
    
    # Get action
    action = controller.get_action(traffic_data)
    print(f"TrafficController initialized successfully!")
    print(f"Recommended action: {action}")