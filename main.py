from ddqn_prb import train_ddqn
import gymnasium as gym
import ale_py
import numpy as np 
import torch
import highway_env
import flappy_bird_gymnasium
import warnings
import random
import os
from fastkan import FastKAN
from discrete_models import get_states_and_actions
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# фиксируем рандом
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

'---------------------------------------------------------'
# выбираем, какие среды запускать для обучения
start_cartPole_mlp = False
start_cartPole_kan = False 

start_acrobot_mlp = False 
start_acrobot_kan = False

start_highway_mlp = False
start_highway_kan = False

start_flappyBird_mlp = False
start_flappyBird_kan = False

start_breakout_mlp = False
start_breakout_kan = False

start_spaceInvaders_mlp = False
start_spaceInvaders_kan = False

# вывод сравнительных графиков после обучения
compare_graphs = True

# вывод графиков после обучения каждой среды
show_results = False # 

# показывать ли картинку сред
render_mode =  None # 'human' 

# сохранить веса модели
save_weights = False

# сохранить результаты обучения
save_results = True

# использовать "уменьшенные" версии сетей
reduced = True
'---------------------------------------------------------'

if reduced:
    import discrete_models_reduced as model
else:   
    import discrete_models as model


if __name__ == "__main__":


    "------------------------------CartPole-----------------------------------------"
    if start_cartPole_mlp:
        print('\n\n\n')
        print('CartPole MLP')
        
        mlp_cartPole = gym.make('CartPole-v1', render_mode=render_mode)    # создаём среду
        state_dim, n_actions = get_states_and_actions(mlp_cartPole)
        model_cartPole_mlp = model.cart_pole_mlp(state_dim, n_actions)    # создаём сеть
        
        mlp_cart_pole_rewards, mlp_cart_pole_losses, mlp_cart_pole_time = train_ddqn(  # единая функция обучения
                model=model_cartPole_mlp, 
                env=mlp_cartPole, 
                target_update_freq=100,
                epsilon_decay=1e4, 
                buffer_capacity=1e2, 
                train_freq=2
        )
        
        if save_results:
            model.save_results(mlp_cartPole, model_cartPole_mlp, mlp_cart_pole_rewards, mlp_cart_pole_losses, mlp_cart_pole_time)



    if start_cartPole_kan:
        print('\n\n\n')
        print('CartPole fastKAN')
        
        kan_cartPole = gym.make('CartPole-v1', render_mode=render_mode)
        state_dim, n_actions = get_states_and_actions(kan_cartPole)
        model_cartPole_kan = model.cartpole_kan(state_dim, n_actions)
        
        kan_cart_pole_rewards, kan_cart_pole_losses, kan_cart_pole_time = train_ddqn(
                model=model_cartPole_kan, 
                env=kan_cartPole, 
                target_update_freq=100,
                epsilon_decay=1e4, 
                buffer_capacity=1e2, 
                train_freq=2
        )
        
        if save_results:
            model.save_results(kan_cartPole, model_cartPole_kan, kan_cart_pole_rewards, kan_cart_pole_losses, kan_cart_pole_time)



    "------------------------------Acrobot-----------------------------------------"
    if start_acrobot_mlp:
        print('\n\n\n')
        print('Acrobot MLP')
        
        mlp_acrobot = gym.make('Acrobot-v1', render_mode=render_mode)
        state_dim, n_actions = get_states_and_actions(mlp_acrobot)
        model_acrobot_mlp = model.acrobot_mlp(state_dim, n_actions)
        
        mlp_acrobot_rewards, mlp_acrobot_losses, mlp_acrobot_time = train_ddqn(
                model=model_acrobot_mlp, 
                env=mlp_acrobot, 
                target_update_freq=1000,
                epsilon_decay=3e4, 
                buffer_capacity=1e3, 
                train_freq=2,
                wanted_score=1
        )
        
        if save_results:
            model.save_results(mlp_acrobot, model_acrobot_mlp, mlp_acrobot_rewards, mlp_acrobot_losses, mlp_acrobot_time)



    if start_acrobot_kan:
        print('\n\n\n')
        print('Acrobot KAN')
        
        kan_acrobot = gym.make('Acrobot-v1', render_mode=render_mode)
        state_dim, n_actions = get_states_and_actions(kan_acrobot)
        model_acrobot_mlp = model.acrobot_kan(state_dim, n_actions)
        
        kan_acrobot_rewards, kan_acrobot_losses, kan_acrobot_time = train_ddqn(
                model=model_acrobot_mlp, 
                env=kan_acrobot, 
                target_update_freq=1000,
                epsilon_decay=3e4, 
                buffer_capacity=1e3, 
                train_freq=2,
                wanted_score=1
        )
        
        if save_results:
            model.save_results(kan_acrobot, model_acrobot_mlp, kan_acrobot_rewards, kan_acrobot_losses, kan_acrobot_time)



    "------------------------------Highway-----------------------------------------"
    if start_highway_mlp:
        print('\n\n\n')
        print('Highway MLP')
        
        mlp_highway = gym.make('highway-v0', render_mode=render_mode)
        state_dim = np.prod(mlp_highway.observation_space.shape) # тут к сожалению только так работает(
        _, n_actions = get_states_and_actions(mlp_highway)
        model_mlp_highway= model.highway_mlp(state_dim, n_actions)
        
        mlp_highway_rewards, mlp_highway_losses, mlp_highway_time = train_ddqn(
                model=model_mlp_highway, 
                env=mlp_highway, 
                target_update_freq=1000, 
                epsilon_decay=3e3, 
                buffer_capacity=1e4, 
                wanted_score=50, 
                train_freq=2
        )
        
        if save_results:
            model.save_results(mlp_highway, model_mlp_highway, mlp_highway_rewards, mlp_highway_losses, mlp_highway_time)



    if start_highway_kan:
        print('\n\n\n')
        print('Highway KAN')
        
        kan_highway = gym.make('highway-v0', render_mode=render_mode)
        state_dim = np.prod(kan_highway.observation_space.shape) 
        _, n_actions = get_states_and_actions(kan_highway)
        model_kan_highway= model.highway_kan(state_dim, n_actions)
        
        kan_highway_rewards, kan_highway_losses, kan_highway_time = train_ddqn(
                model=model_kan_highway, 
                env=kan_highway, 
                target_update_freq=1000, 
                epsilon_decay=3e3, 
                buffer_capacity=1e4, 
                wanted_score=50, 
                train_freq=2
        )
        
        if save_results:
            model.save_results(kan_highway, model_kan_highway, kan_highway_rewards, kan_highway_losses, kan_highway_time)



    "------------------------------FlappyBird-----------------------------------------"
    if start_flappyBird_mlp:
        print('\n\n\n')
        print('FlappyBird MLP')
        
        mlp_flappy_bird = gym.make("FlappyBird-v0", use_lidar=True, render_mode=render_mode) 
        state_dim, n_actions = get_states_and_actions(mlp_flappy_bird)
        model_flappyBird_mlp = model.flappyBird_mlp(state_dim, n_actions)
        
        mlp_flappy_bird_rewards, mlp_flappy_bird_losses, mlp_flappy_bird_time = train_ddqn(
                model=model_flappyBird_mlp, 
                env=mlp_flappy_bird,  
                target_update_freq=1000, 
                epsilon_decay=5e3, 
                buffer_capacity=1e4, 
                wanted_score=50,
                train_freq=2
        )
        
        if save_results:
            model.save_results(mlp_flappy_bird, model_flappyBird_mlp, mlp_flappy_bird_rewards, mlp_flappy_bird_losses, mlp_flappy_bird_time)



    if start_flappyBird_kan:
        print('\n\n\n')
        print('FlappyBird KAN')
        
        kan_flappy_bird = gym.make("FlappyBird-v0", use_lidar=True, render_mode=render_mode) 
        state_dim, n_actions = get_states_and_actions(kan_flappy_bird)
        model_flappyBird_kan = model.flappyBird_kan(state_dim, n_actions)
        
        kan_flappy_bird_rewards, kan_flappy_bird_losses, kan_flappy_bird_time = train_ddqn(
                model=model_flappyBird_kan, 
                env=kan_flappy_bird, 
                target_update_freq=1000, 
                epsilon_decay=5e3, 
                buffer_capacity=1e4, 
                wanted_score=50,
                train_freq=2
        )
        
        if save_results:
            model.save_results(kan_flappy_bird, model_flappyBird_kan, kan_flappy_bird_rewards, kan_flappy_bird_losses, kan_flappy_bird_time)



    "------------------------------Breakout-----------------------------------------"
    if start_breakout_mlp:
        print('\n\n\n')
        print('Breakout MLP')
        
        gym.register_envs(ale_py)
        mlp_breakout = gym.make('ALE/Breakout-ram-v5', render_mode=render_mode)   # ALE/Pong-ram-v5
        state_dim, n_actions = get_states_and_actions(mlp_breakout)
        model_mlp_porg = model.breakout_mlp(state_dim, n_actions)
        
        mlp_breakout_rewards, mlp_breakout_losses, mlp_breakout_time = train_ddqn(
                model=model_mlp_porg, 
                env=mlp_breakout, 
                target_update_freq=1000, 
                epsilon_decay=4e4, 
                buffer_capacity=1e4, 
                wanted_score=500,
                train_freq=2
        )
        
        if save_results:
            model.save_results(mlp_breakout, model_mlp_porg, mlp_breakout_rewards, mlp_breakout_losses, mlp_breakout_time)



    if start_breakout_kan:
        print('\n\n\n')
        print('breakout fastKAN')
        
        gym.register_envs(ale_py)
        kan_breakout = gym.make('ALE/Breakout-ram-v5', render_mode=render_mode)     
        state_dim, n_actions = get_states_and_actions(kan_breakout)
        model_breakout_kan = model.breakout_kan(state_dim, n_actions)

        kan_breakout_rewards, kan_breakout_losses, kan_breakout_time = train_ddqn(
                model=model_breakout_kan, 
                env=kan_breakout, 
                target_update_freq=1000, 
                epsilon_decay=4e4, 
                buffer_capacity=1e4, 
                wanted_score=500,
                train_freq=2
        )
        
        if save_results:
            model.save_results(kan_breakout, model_breakout_kan, kan_breakout_rewards, kan_breakout_losses, kan_breakout_time)



    "------------------------------SpaceInvaders-----------------------------------------"
    if start_spaceInvaders_mlp:
        print('\n\n\n')
        print('SpaceInvaders MLP')
        
        gym.register_envs(ale_py)
        mlp_spaceInvaders = gym.make('ALE/SpaceInvaders-ram-v5', render_mode=render_mode)
        state_dim, n_actions = get_states_and_actions(mlp_spaceInvaders)
        model_mlp_spaceInvaders = model.spaceInvaders_mlp(state_dim, n_actions)
        
        mlp_SpaceInvaders_rewards, mlp_SpaceInvaders_losses, mlp_SpaceInvaders_time = train_ddqn(
                model=model_mlp_spaceInvaders, 
                env=mlp_spaceInvaders, 
                target_update_freq=1000, 
                epsilon_decay=5e5, 
                buffer_capacity=1e4, 
                wanted_score=1000, 
                train_freq=2
            )
        if save_results:
            model.save_results(mlp_spaceInvaders, model_mlp_spaceInvaders, mlp_SpaceInvaders_rewards, mlp_SpaceInvaders_losses, mlp_SpaceInvaders_time)



    if start_spaceInvaders_kan:
        print('\n\n\n')
        print('SpaceInvaders KAN')
        
        gym.register_envs(ale_py)
        kan_spaceInvaders = gym.make('ALE/SpaceInvaders-ram-v5', render_mode=render_mode)
        state_dim, n_actions = get_states_and_actions(kan_spaceInvaders)
        model_kan_spaceInvaders = model.spaceInvaders_kan(state_dim, n_actions)

        kan_spaceInvaders_rewards, kan_spaceInvaders_losses, kan_spaceInvaders_time = train_ddqn(
                model=model_kan_spaceInvaders, 
                env=kan_spaceInvaders, 
                target_update_freq=1000, 
                epsilon_decay=5e5, 
                buffer_capacity=1e4, 
                wanted_score=1000, 
                train_freq=2
        )
        
        if save_results:
            model.save_results(kan_spaceInvaders, model_kan_spaceInvaders, kan_spaceInvaders_rewards, kan_spaceInvaders_losses, kan_spaceInvaders_time)


    "-----------------------------------------------------------------------"
    moving_mean = 20  # используем скользящее среднее для вывода графиков
    if compare_graphs:
        if reduced:
            model.compare("results/cart_pole_mlp-reduced-CartPole-v1", "results/FastKAN-reduced-CartPole-v1", "CartPole-reduced", moving_mean)
            model.compare("results/acrobot_mlp-reduced-Acrobot-v1", "results/FastKAN-reduced-Acrobot-v1", "Acrobot-reduced", moving_mean)
            model.compare("results/highway_mlp-reduced-highway-v0", "results/FastKAN-reduced-highway-v0", "Highway-reduced", moving_mean)
            model.compare("results/flappyBird_mlp-reduced-FlappyBird-v0", "results/FastKAN-reduced-FlappyBird-v0", "FlappyBird-reduced", moving_mean)
            model.compare("results/breakout_mlp-reduced-Breakout-ram-v5", "results/FastKAN-reduced-Breakout-ram-v5", "breakout-reduced", moving_mean)
            model.compare("results/spaceInvaders_mlp-reduced-SpaceInvaders-ram-v5", "results/FastKAN-reduced-SpaceInvaders-ram-v5", "SpaceInvaders-reduced", moving_mean)

        else:
            model.compare("results/cart_pole_mlp-CartPole-v1", "results/FastKAN-CartPole-v1", "CartPole", moving_mean)
            model.compare("results/acrobot_mlp-Acrobot-v1", "results/FastKAN-Acrobot-v1", "Acrobot", moving_mean)
            model.compare("results/highway_mlp-highway-v0", "results/FastKAN-highway-v0", "Highway", moving_mean)
            model.compare("results/flappyBird_mlp-FlappyBird-v0", "results/FastKAN-FlappyBird-v0", "FlappyBird", moving_mean)
            model.compare("results/breakout_mlp-Breakout-ram-v5", "results/FastKAN-Breakout-ram-v5", "breakout", moving_mean)
            model.compare("results/spaceInvaders_mlp-SpaceInvaders-ram-v5", "results/FastKAN-SpaceInvaders-ram-v5", "SpaceInvaders", moving_mean)

