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
    parser.add_argument('--n-epochs', type=int, default=20000, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=200, help='the times to update the network')
    parser.add_argument('--seed', type=int, default=123, help='random seed')

    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy of low-level')
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
    parser.add_argument('--action-l2', type=float, default=0.5, help='l2 reg')

    parser.add_argument('--metric', type=str, default='MLP', help='the metric for the distance embedding')
    parser.add_argument('--device', type=str, default="cuda:1", help='cuda device')

    parser.add_argument('--lr-decay-actor', type=int, default=3000, help='actor learning rate decay')
    parser.add_argument('--lr-decay-critic', type=int, default=4000, help='critic learning rate decay')
    parser.add_argument('--layer', type=int, default=6, help='number of layers for critic')

    parser.add_argument('--period', type=int, default=10, help='target update period')
    parser.add_argument('--distance', type=float, default=0.1, help='distance threshold for HER')

    parser.add_argument('--resume', type=bool, default=False, help='resume or not')
    # Will be considered only if resume is True
    parser.add_argument('--resume-epoch', type=int, default=0, help='resume epoch')
    parser.add_argument('--resume-path', type=str,
                        default='saved_models/PointMaze1-v1_May26_11-11-42',
                        help='resume path')

    # hier para
    parser.add_argument('--save', type=bool, default=False, help='save model and tensorboard data')
    parser.add_argument('--animate', type=bool, default=True)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--xy_goal", type=bool, default=True, help="use xy position as goal in create_env")
    parser.add_argument('--eval_interval', type=int, default=50, help="every n episodes to eval once")
    parser.add_argument('--c', type=int, default=10, help="interval of high-level action")
    parser.add_argument('--contrastive_phi', type=bool, default=True, help='learn feature with contrastive loss')

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

    # hier para
    parser.add_argument('--save', type=bool, default=True, help='save model and tensorboard data')
    parser.add_argument('--animate', type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument('--eval_interval', type=int, default=100, help="every n episodes to eval once")
    parser.add_argument('--ebp', type=bool, default=False, help='energy based prioritized replay')
    parser.add_argument('--c', type=int, default=25, help="interval of high-level action")
    parser.add_argument('--hi_random_eps', type=float, default=0.2, help="prob of taking nearby subgoal")
    parser.add_argument('--hi_n_batches', type=int, default=5, help='the times to update high-level network')
    parser.add_argument('--hi_replay_strategy', type=str, default='future', help='the HER strategy of high-level')
    parser.add_argument('--hi_lr_actor', type=float, default=0.00002, help='the learning rate of the actor')
    parser.add_argument('--contrastive_phi', type=bool, default=False, help='learn feature with contrastive loss')

    args = parser.parse_args()
    return args

def get_args_ant():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='AntMaze1-v1', help='the environment name')
    parser.add_argument('--test', type=str, default='AntMaze1Test-v1')
    parser.add_argument('--n-epochs', type=int, default=14000, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=200, help='the times to update the network')
    parser.add_argument('--seed', type=int, default=256, help='random seed')

    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise factor for Gaussian')
    parser.add_argument('--random-eps', type=float, default=0.2, help="prob for acting randomly")

    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=5, help='ratio to be replaced')
    parser.add_argument('--future-step', type=int, default=200, help='future step to be sampled')
    parser.add_argument('--batch-size', type=int, default=128, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=0., help='l2 reg')
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
    parser.add_argument('--resume-path', type=str, default='saved_models/AntMaze1-v1_May18_15-09-32', help='resume path')

    # add for hier policy
    parser.add_argument('--save', type=bool, default=True, help='save model and tensorboard data')
    parser.add_argument('--animate', type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--xy_goal", type=bool, default=True, help="use only two dimension goal space for ant env")
    parser.add_argument('--eval_interval', type=int, default=50, help="every n episodes to eval once")
    parser.add_argument('--c', type=int, default=20, help="interval of high-level action")
    parser.add_argument('--contrastive_phi', type=bool, default=False, help='learn feature with contrastive loss')
    parser.add_argument('--hi_dim', type=int, default=0, help="dim of high-level obs, if 0, use full obs")
    parser.add_argument('--low_dim', type=int, default=0, help="dim of low-level obs, if 0, use full obs")

    # args of sac
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                    term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--start_epoch', type=int, default=300, metavar='N',
                        help='Epochs sampling random actions (default: 50)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')

    args = parser.parse_args()
    return args
