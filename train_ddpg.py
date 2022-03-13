import numpy as np
import gym
from arguments.arguments_ddpg import get_args_ant, get_args_chain, get_args
from algos.ddpg_agent import ddpg_agent
from algos.ddpg_explore import ddpg_explore
from goal_env.mujoco import *
from goal_env.robotics import *
import random
import torch


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def launch(args):
    # create the ddpg_agent
    env = gym.make(args.env_name)
    test_env = gym.make(args.test)
    # # add noise
    # env.env.env.wrapped_env.add_noise = args.add_noise
    # test_env.env.env.wrapped_env.add_noise = args.add_noise
    # set random seeds for reproduce
    env.seed(args.seed)
    if args.env_name != "NChain-v1":
        env.env.env.wrapped_env.seed(args.seed)
        test_env.env.env.wrapped_env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device is not 'cpu':
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    if args.env_name[:3] in ["Ant", "Poi"]:
        env.env.env.visualize_goal = args.animate
        test_env.env.env.visualize_goal = args.animate
    env_params = get_env_params(env)
    env_params['max_test_timesteps'] = test_env._max_episode_steps
    # create the ddpg agent to interact with the environment
    ddpg_trainer = ddpg_agent(args, env, env_params, test_env)
    if args.eval:
        # ddpg_trainer._eval_agent()
        # ddpg_trainer.visualize_representation(100)
        ddpg_trainer.multi_eval()
    else:
        ddpg_trainer.learn()


# get the params
args = get_args_ant()
# args = get_args_chain()
# args = get_args()  # point
if __name__ == '__main__':
    launch(args)
