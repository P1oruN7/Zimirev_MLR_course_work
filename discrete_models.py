import torch.nn as nn
from fastkan import FastKAN
import json
import matplotlib.pyplot as plt
import pandas as pd

# тут задаются модели и полезные функции, типа сохранения, получения, визуализации

"------------------------------CartPole-----------------------------------------"
class cart_pole_mlp(nn.Module):                               # params 17026
    def __init__(self, state_dim, n_actions):
        super(cart_pole_mlp, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def cartpole_kan(state_dim, n_actions):
    return FastKAN([state_dim, 28, 58, n_actions], num_grids=8)  # params 16960



"------------------------------Acrobot-----------------------------------------"
class acrobot_mlp(nn.Module):                     # params 17219
    def __init__(self, state_dim, n_actions):
        super(acrobot_mlp, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def acrobot_kan(state_dim, n_actions):
    return FastKAN([state_dim, 28, 56, n_actions], num_grids=8)       # params 17427



"------------------------------Highway-----------------------------------------"
class highway_mlp(nn.Module):                               # params 69893
    def __init__(self, state_dim, n_actions):
        super(highway_mlp, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
    
def highway_kan(state_dim, n_actions):
    return FastKAN([state_dim, 74, 74, n_actions], num_grids=8)      # params 69787



"------------------------------FlappyBird-----------------------------------------"
class flappyBird_mlp(nn.Module):                                    # params 56450
    def __init__(self, state_dim, n_actions):
        super(flappyBird_mlp, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def flappyBird_kan(state_dim, n_actions):
    return FastKAN([state_dim, 36, 38, n_actions], num_grids=6)         # params 56070



"------------------------------Breakout-----------------------------------------"
class breakout_mlp(nn.Module):                                      # params 182148
    def __init__(self, state_dim, n_actions):
        super(breakout_mlp, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def breakout_kan(state_dim, n_actions):
    return FastKAN([state_dim, 104, 104, 88, 64,  n_actions], num_grids=8)  # params 182850



"------------------------------SpaceInvaders-----------------------------------------"
class spaceInvaders_mlp(nn.Module):                                        # params 182662
    def __init__(self, state_dim, n_actions):
        super(spaceInvaders_mlp, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def spaceInvaders_kan(state_dim, n_actions):
    return FastKAN([state_dim, 128, 104, 88, 64,  n_actions], num_grids=8)    # params 182417



"------------------------------Utils-----------------------------------------"
def get_states_and_actions(env):
    states = env.observation_space.shape[0]
    actions = env.action_space.n
    return states, actions

def save_results (env, model, rewards, losses, time):               
    params = sum(p.numel() for p in model.parameters())
    env_name = env.spec.id
    env_name = env_name.replace("ALE/", "")
    model_name = model.__class__.__name__
    data = {"rewards": rewards,
            "losses": losses,
            "time": time,
            "params":params}
    with open(f'results/{model_name}-{env_name}.json', 'w') as f:
        json.dump(data, f)

def get_results (file_name):
    with open(f'{file_name}.json', 'r') as f:
        loaded_data = json.load(f)

    rewards = loaded_data['rewards']
    losses = loaded_data['losses']
    time = loaded_data['time']
    params = loaded_data['params']

    return rewards, losses, time, params


def compare(mlp_file_name, kan_file_name, title, window=3):
    mlp_rewards, mlp_losses, mlp_time, mlp_params = get_results(mlp_file_name)
    kan_rewards, kan_losses, kan_time, kan_params = get_results(kan_file_name)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    smoothed_mlp_rewards = pd.Series(mlp_rewards).rolling(window=window, min_periods=1).mean() # 
    smoothed_kan_rewards = pd.Series(kan_rewards).rolling(window=window, min_periods=1).mean()
    
    # График наград
    ax1.set_title(title)
    ax1.plot(smoothed_mlp_rewards, label='MLP Rewards', color='blue')
    ax1.plot(smoothed_kan_rewards, label='KAN Rewards', color='red')
    ax1.set_xlabel("Эпизод")
    ax1.set_ylabel("Награда")
    ax1.legend()
    ax1.grid()
    
    # График лоссов
    ax2.set_title("Сравнение потерь моделей MLP и KAN")
    ax2.plot(mlp_losses, label='MLP Loss', color='blue')
    ax2.plot(kan_losses, label='KAN Loss', color='red')
    ax2.set_xlabel("Эпизод")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid()
    
    # Вывод параметров и времени обучения
    plt.figtext(0.5, 0.05, f"MLP Время обучения: {mlp_time:.2f} сек на 100 эпох | Параметры: {mlp_params}", fontsize=12, ha='center')
    plt.figtext(0.5, 0.03, f"KAN Время обучения: {kan_time:.2f} сек на 100 эпох | Параметры: {kan_params}", fontsize=12, ha='center')
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()

    print(title)
    print(f"Время обучения:    MLP = {mlp_time:.2f}    KAN = {kan_time:.2f}")
    print(f"Количество параметров:    MLP = {mlp_params}    KAN = {kan_params}")
    print(f"Сумма наград:    MLP = {sum(mlp_rewards)}    KAN = {sum(kan_rewards)}")
    print(f"Средняя награда за эпизод:    MLP = {sum(mlp_rewards)/ len(mlp_rewards):.2f}    KAN = {sum(kan_rewards)/ len(kan_rewards):.2f}")