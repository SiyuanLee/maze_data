import argparse

"""
Here are the param for the training

"""


# args for point maze
def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='PointMaze1-v1', help='the environment name')
    parser.add_argument('--test', type=str, default='PointMaze1Test-v1')
    parser.add_argument('--n-epochs', type=int, default=8000, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=200, help='the times to update the network')
    parser.add_argument('--seed', type=int, default=125, help='random seed')

    parser.add_argument('--replay-strategy', type=str, default='none', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise factor for Gaussian')
    parser.add_argument('--random-eps', type=float, default=0.2, help="prob for acting randomly")

    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=5, help='ratio to be replaced')
    parser.add_argument('--future-step', type=int, default=80, help='future step to be sampled')
    parser.add_argument('--batch-size', type=int, default=128, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--lr-actor', type=float, default=0.0002, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.0002, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.99, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--action-l2', type=float, default=0.5, help='l2 reg')

    parser.add_argument('--metric', type=str, default='MLP', help='the metric for the distance embedding')
    parser.add_argument('--device', type=str, default="cuda:4", help='cuda device')

    parser.add_argument('--lr-decay-actor', type=int, default=3000, help='actor learning rate decay')
    parser.add_argument('--lr-decay-critic', type=int, default=4000, help='critic learning rate decay')
    parser.add_argument('--layer', type=int, default=6, help='number of layers for critic')

    parser.add_argument('--period', type=int, default=10, help='target update period')
    parser.add_argument('--distance', type=float, default=0.1, help='distance threshold for HER')

    parser.add_argument('--resume', type=bool, default=False, help='resume or not')
    # Will be considered only if resume is True
    parser.add_argument('--resume-epoch', type=int, default=10000, help='resume epoch')
    parser.add_argument('--resume-path', type=str,
                        default='saved_models/PointMaze1-v1_May09_17-00-16',
                        help='resume path')

    # add for explore
    parser.add_argument('--save', type=bool, default=True, help='save model and tensorboard data')
    parser.add_argument('--animate', type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--xy_goal", type=bool, default=True, help="use only two dimension goal space for ant env")
    parser.add_argument('--eval_interval', type=int, default=50, help="every n episodes to eval once")

    args = parser.parse_args()
    return args


def get_args_ant():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='AntMaze1-v1', help='the environment name')
    parser.add_argument('--test', type=str, default='AntMaze1Test-v1')
    parser.add_argument('--n-epochs', type=int, default=9000, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=200, help='the times to update the network')
    parser.add_argument('--seed', type=int, default=7, help='random seed')

    parser.add_argument('--replay-strategy', type=str, default='none', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise factor for Gaussian')
    parser.add_argument('--random-eps', type=float, default=0.2, help="prob for acting randomly")

    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=5, help='ratio to be replaced')
    parser.add_argument('--future-step', type=int, default=200, help='future step to be sampled')
    parser.add_argument('--batch-size', type=int, default=128, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=0.5, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.0002, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.0002, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.99, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')

    parser.add_argument('--metric', type=str, default='MLP', help='the metric for the distance embedding')
    parser.add_argument('--device', type=str, default="cuda:0", help='cuda device')

    parser.add_argument('--lr-decay-actor', type=int, default=3000, help='actor learning rate decay')
    parser.add_argument('--lr-decay-critic', type=int, default=3000, help='critic learning rate decay')
    parser.add_argument('--layer', type=int, default=6, help='number of layers for critic')

    parser.add_argument('--period', type=int, default=3, help='target update period')
    parser.add_argument('--distance', type=float, default=0.1, help='distance threshold for HER')

    parser.add_argument('--resume', type=bool, default=False, help='resume or not')
    # Will be considered only if resume is True
    parser.add_argument('--resume-epoch', type=int, default=500, help='resume epoch')
    parser.add_argument('--resume-path', type=str, default='saved_models/AntMaze1-v1_Nov23_22-40-31', help='resume path')

    # add for explore
    parser.add_argument('--save', type=bool, default=True, help='save model and tensorboard data')
    parser.add_argument('--animate', type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument('--eval_interval', type=int, default=50, help="every n episodes to eval once")
    parser.add_argument("--image", type=bool, default=False, help='use image input')
    parser.add_argument('--random_start', type=int, default=1, help='1: random in the upper row; 2: lower corner; 0: farthest point')

    args = parser.parse_args()
    return args


def get_args_chain():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='NChain-v1', help='the environment name')
    parser.add_argument('--test', type=str, default='NChain-v1')
    parser.add_argument('--n-epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=200, help='the times to update the network')
    parser.add_argument('--seed', type=int, default=263, help='random seed')

    parser.add_argument('--replay-strategy', type=str, default='none', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise factor for Gaussian')
    parser.add_argument('--random-eps', type=float, default=1.0, help="prob for acting randomly")

    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=5, help='ratio to be replaced')
    parser.add_argument('--future-step', type=int, default=200, help='future step to be sampled')
    parser.add_argument('--batch-size', type=int, default=128, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=0.0, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.0002, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.0002, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.99, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')

    parser.add_argument('--metric', type=str, default='MLP', help='the metric for the distance embedding')
    parser.add_argument('--device', type=str, default="cuda:2", help='cuda device')

    parser.add_argument('--lr-decay-actor', type=int, default=3000, help='actor learning rate decay')
    parser.add_argument('--lr-decay-critic', type=int, default=3000, help='critic learning rate decay')
    parser.add_argument('--layer', type=int, default=6, help='number of layers for critic')

    parser.add_argument('--period', type=int, default=3, help='target update period')
    parser.add_argument('--distance', type=float, default=0.1, help='distance threshold for HER')

    parser.add_argument('--resume', type=bool, default=False, help='resume or not')
    # Will be considered only if resume is True
    parser.add_argument('--resume-epoch', type=int, default=500, help='resume epoch')
    parser.add_argument('--resume-path', type=str, default='saved_models/AntMaze-v1_Apr20_10-58-41', help='resume path')

    # add for explore
    parser.add_argument('--save', type=bool, default=True, help='save model and tensorboard data')
    parser.add_argument('--animate', type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument('--eval_interval', type=int, default=50, help="every n episodes to eval once")
    # parser.add_argument('--add_noise', type=bool, default=True, help='add 10 dim noise to obs')

    args = parser.parse_args()
    return args


def get_args_fetch():
    parser = argparse.ArgumentParser()
    # the environment setting, v3 is original env, v2 is modified env
    parser.add_argument('--env-name', type=str, default='FetchPush-v3', help='the environment name')
    parser.add_argument('--test', type=str, default='FetchPush-v3')
    parser.add_argument('--n-epochs', type=int, default=35000, help='the number of episodes to train the agent')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--seed', type=int, default=400, help='random seed')

    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise factor for Gaussian')
    parser.add_argument('--random-eps', type=float, default=0.3, help="prob for acting randomly")

    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replaced')
    parser.add_argument('--future-step', type=int, default=40, help='future step to be sampled')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1.0, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')

    parser.add_argument('--metric', type=str, default='MLP', help='the metric for the distance embedding')
    parser.add_argument('--device', type=str, default="cuda:1", help='cuda device')

    parser.add_argument('--lr-decay-actor', type=int, default=30000, help='actor learning rate decay')
    parser.add_argument('--lr-decay-critic', type=int, default=30000, help='critic learning rate decay')
    parser.add_argument('--layer', type=int, default=4, help='number of layers for critic')

    parser.add_argument('--period', type=int, default=3, help='target update period')
    parser.add_argument('--distance', type=float, default=0.05, help='distance threshold for HER')

    parser.add_argument('--resume', type=bool, default=False, help='resume or not')
    # Will be considered only if resume is True
    parser.add_argument('--resume-epoch', type=int, default=2500, help='resume epoch')
    parser.add_argument('--resume-path', type=str, default='saved_models/FetchPush-v0_Apr28_22-04-49',
                        help='resume path')

    # add for explore
    parser.add_argument('--explore', type=bool, default=False, help='if use frontier exploration')
    parser.add_argument('--save', type=bool, default=False, help='save model and tensorboard data')
    parser.add_argument('--animate', type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument('--eval_interval', type=int, default=100, help="every n episodes to eval once")
    parser.add_argument('--ebp', type=bool, default=True, help='energy based prioritized replay')

    args = parser.parse_args()
    return args
