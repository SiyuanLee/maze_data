import argparse

"""
Here are the param for the training

"""


# args for point maze
def get_args_point():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='PointMaze1-v1', help='the environment name')
    parser.add_argument('--test', type=str, default='PointMaze1Test-v1')
    parser.add_argument('--n-epochs', type=int, default=8000, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=200, help='the times to update the network')
    parser.add_argument('--seed', type=int, default=134, help='random seed')

    parser.add_argument('--replay-strategy', type=str, default='none', help='the HER strategy of low-level')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise factor for Gaussian')
    parser.add_argument('--random-eps', type=float, default=0.2, help="prob for acting randomly of low-level")

    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=5, help='ratio to be replaced')
    parser.add_argument('--future-step', type=int, default=80, help='future step to be sampled')
    parser.add_argument('--batch-size', type=int, default=128, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--lr-actor', type=float, default=0.0002, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.0002, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.99, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--action-l2', type=float, default=0.0, help='l2 reg')

    parser.add_argument('--metric', type=str, default='MLP', help='the metric for the distance embedding')
    parser.add_argument('--device', type=str, default="cuda:6", help='cuda device')

    parser.add_argument('--lr-decay-actor', type=int, default=3000, help='actor learning rate decay')
    parser.add_argument('--lr-decay-critic', type=int, default=4000, help='critic learning rate decay')
    parser.add_argument('--layer', type=int, default=6, help='number of layers for critic')

    parser.add_argument('--period', type=int, default=10, help='target update period')
    parser.add_argument('--distance', type=float, default=0.1, help='distance threshold for HER')

    parser.add_argument('--resume', type=bool, default=False, help='resume or not')
    # Will be considered only if resume is True
    parser.add_argument('--resume-epoch', type=int, default=0, help='resume epoch')
    parser.add_argument('--resume-path', type=str,
                        default='saved_models/PointMaze1-v1_Jul11_11-12-48',
                        help='resume path')

    # add for hier policy
    parser.add_argument('--save', type=bool, default=True, help='save model and tensorboard data')
    parser.add_argument('--animate', type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument('--eval_interval', type=int, default=50, help="every n episodes to eval once")
    parser.add_argument('--low_reward_coeff', type=float, default=1.0, help='low-level reward coeff')


    # args of sac
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')

    args = parser.parse_args()
    return args


def get_args_ant():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='AntPush-v1', help='the environment name')
    parser.add_argument('--test', type=str, default='AntPushTest-v1')
    parser.add_argument('--n-epochs', type=int, default=8000, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=200, help='the times to update the network')
    parser.add_argument('--seed', type=int, default=271, help='random seed')

    parser.add_argument('--replay-strategy', type=str, default='none', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise factor for Gaussian')
    parser.add_argument('--random-eps', type=float, default=0.2, help="prob for acting randomly")

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
    parser.add_argument('--device', type=str, default="cuda:3", help='cuda device')

    parser.add_argument('--lr-decay-actor', type=int, default=3000, help='actor learning rate decay')
    parser.add_argument('--lr-decay-critic', type=int, default=3000, help='critic learning rate decay')
    parser.add_argument('--layer', type=int, default=6, help='number of layers for critic')

    parser.add_argument('--period', type=int, default=3, help='target update period')
    parser.add_argument('--distance', type=float, default=0.1, help='distance threshold for HER')

    parser.add_argument('--resume', type=bool, default=False, help='resume or not')
    # Will be considered only if resume is True
    parser.add_argument('--resume-epoch', type=int, default=0, help='resume epoch')
    parser.add_argument('--resume-path', type=str, default='saved_models/AntFall-v1_Jul10_14-20-07', help='resume path')

    # add for hier policy
    parser.add_argument('--save', type=bool, default=True, help='save model and tensorboard data')
    parser.add_argument('--animate', type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument('--eval_interval', type=int, default=50, help="every n episodes to eval once")
    parser.add_argument('--low_reward_coeff', type=float, default=1.0, help='low-level reward coeff')
    parser.add_argument("--image", type=bool, default=True, help='use image input')

    # args of sac
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')

    args = parser.parse_args()
    return args


