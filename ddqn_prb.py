import time 
import copy 
import random 
from collections import deque 
import gymnasium as gym 
import numpy as np 
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm 
from datetime import datetime 

# Используем Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# единая функция обучения double DQN
def train_ddqn(model,env,
               lr=1e-4, loss_fcn = torch.nn.HuberLoss(reduction='none'), 
               num_episodes=2000, 
               batch_size=32, 
               gamma=0.99, 
               target_update_freq=1000, 
               epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500, 
               buffer_capacity=10000, 
               save_weights=False, file_prefix = '', 
               show_results=True, wanted_score = 1000, 
               train_freq = 1):
    
    from main import save_weights, show_results   # знаю, что циклические импорты не очень хорошо, но тут слишком удобно

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Количество параметров модели: {num_params}")

    if (not file_prefix) and (save_weights):
        file_prefix = model.__class__.__name__

    device = torch.device('cpu')  # Эксперементировал с обучением на gpu, но так получалось только медленнее
    #print(device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    model.to(device)

    target_model = copy.deepcopy(model)
    target_model.to(device)
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    replay_buffer = PrioritizedReplayBuffer(int(buffer_capacity), alpha=0.6)


    episode_rewards = []
    losses_history = []
    
    total_steps = 0
    training_time = 0.0

    beta_start = 0.2
    beta_frames = num_episodes * 100  # оценка общего числа шагов, взял какое-то примерно среднее значение

    for episode in tqdm(range(1, num_episodes + 1), desc="Episodes"):            
        state, _ = env.reset()  ###
        state = np.array(state, dtype=np.float32).flatten()

        episode_reward = 0
        episode_losses = []
        done = False

        while not done:
            total_steps += 1
            # Нашёл в интернете такую функцию для снижения эпсилон, вроде лучше, чем линейная
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * total_steps / epsilon_decay)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32).flatten()
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if (episode_reward >= wanted_score) and (wanted_score>0):
                done = True

            if (len(replay_buffer) >= batch_size) and (total_steps % train_freq == 0):
                update_start = time.time()  

                beta = min(1.0, beta_start + total_steps * (1.0 - beta_start) / beta_frames)

                states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)
                states_tensor = torch.FloatTensor(states).to(device)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states_tensor = torch.FloatTensor(next_states).to(device)
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(device)
                weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(device)

                current_q_values = model(states_tensor).gather(1, actions_tensor)
                with torch.no_grad():
                    next_actions = model(next_states_tensor).argmax(1, keepdim=True)
                    next_q_values = target_model(next_states_tensor).gather(1, next_actions)
                    target_q_values = rewards_tensor + (1 - dones_tensor) * gamma * next_q_values

                loss_elements = loss_fcn(current_q_values, target_q_values)
                loss = (loss_elements * weights_tensor).mean()

                optimizer.zero_grad()
                loss.backward()
                # Применяем gradient clipping (ограничение градиентов по norm). С ним работает чуть лучше
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                episode_losses.append(loss.item())
                training_time += time.time() - update_start

                td_errors = (current_q_values - target_q_values).detach().abs().cpu().numpy().squeeze()
                new_priorities = td_errors + 1e-5 # небольшое смещение, чтобы не было нулевых приоритетов
                replay_buffer.update_priorities(indices, new_priorities)

            if total_steps % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

        episode_rewards.append(episode_reward)
        mean_loss = np.mean(episode_losses) if episode_losses else 0
        losses_history.append(mean_loss)

        tqdm.write(f"Episode: {episode:3d} | Reward: {episode_reward:7.2f} | ε: {epsilon:5.3f} | Loss: {mean_loss:7.4f} | Steps: {total_steps:6d}")

    training_time_norm = training_time/num_episodes*100  # время обучения на 100 эпизодов. Изначально планировал разное количество эпизодов для разных сред
    params = sum(p.numel() for p in model.parameters())

    if save_weights:
        formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        file_name = f"{file_prefix}-{formatted_time}.pth"
        torch.save(model.state_dict(), file_name)
        print(f"Веса модели сохранены в {file_name}")
    
    torch.cuda.empty_cache()
    
    if show_results:
        import pandas as pd
        smoothed_rewards = pd.Series(episode_rewards).rolling(window=3, min_periods=1).mean()

        env_name = env.spec.id

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.set_title(f"Награды модели {model.__class__.__name__} в среде {env_name} ")
        ax1.plot(smoothed_rewards, label='Награды')
        ax1.axhline(y=wanted_score, color='red', linestyle='--')
        ax1.set_xlabel("Эпизод")
        ax1.set_ylabel("Награда")
        ax1.tick_params(axis='y')
        ax1.grid()

        ax2.set_title(f"Loss модели {model.__class__.__name__} в среде {env_name} ")
        ax2.plot(losses_history, label='Loss')
        ax2.set_xlabel("Эпизод")
        ax2.set_ylabel("Loss")
        ax2.tick_params(axis='y')
        ax2.grid()

        plt.figtext(0.5, 0.05, f"Время обучения: {training_time_norm:.2f} сек на 100 эпизодов", fontsize=12, ha='center')
        plt.figtext(0.5, 0.03, f"Параметры: {params}", fontsize=12, ha='center')

        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()
    
    return episode_rewards, losses_history, training_time_norm
