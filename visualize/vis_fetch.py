import torch
torch.backends.cudnn.benchmark = True
import time
import sys
sys.path.append('../')
from arguments_ddpg import get_args_fetch
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
sns.set_style("whitegrid", {'axes.grid': False})
from goal_env.mujoco.maze_env_utils import construct_maze
from planner.explore_goal_plan_count import *
from algos.ddpg_explore import ddpg_explore
import gym
from train_ddpg import get_env_params
from planner.simhash import HashingBonusEvaluator
import matplotlib.patches as patches


def extract_mapping(resume_path, show=True):
    buffer = torch.load(resume_path + '/replaybuffer.pt')
    landmarks = buffer.get_all_data()['ag']
    if show:
        # plot_map()
        landmarks = landmarks.reshape(-1, landmarks.shape[2])
        # landmarks = np.concatenate(landmarks)
        print("shape", landmarks.shape)
        plt.scatter(landmarks.T[0], landmarks.T[1], color='blue', s=5)
        plt.show()
    return buffer, landmarks

def count_frontier(landmarks):
    dim_key = 128
    goal_dim = 3
    hash = HashingBonusEvaluator(dim_key, goal_dim)
    print("dim_key", dim_key)
    hash.inc_hash(landmarks[:, :goal_dim])
    count = hash.predict(landmarks[:, :goal_dim])
    num = 50

    start_time = time.time()
    idxs = np.argpartition(count, num)[:num]
    print("use_time", time.time() - start_time)
    print("k", count[idxs])

    frontier = landmarks[idxs]
    plt.scatter(landmarks.T[0], landmarks.T[1], color='blue', s=5)
    plt.scatter(frontier.T[0], frontier.T[1], color='red', s=5)

    currentAxis = plt.gca()
    rect = patches.Rectangle((1., 0.4), 0.6, 0.7, linewidth=3, edgecolor='k', facecolor='none')
    currentAxis.add_patch(rect)

    points = [[1, 0.4],
              [1.6, 0.4],
              [1.6, 1.1],
              [1.0, 1.1],
              [1, 0.4]]
    # points = *points
    plt.plot(*points[0], *points[1], color="k")

    plt.show()




def plot_walls(walls, scaling, row_r, col_r):
    for (i, j) in zip(*np.where(walls)):
        i = i * scaling
        j = j * scaling
        x = np.array([j, j + scaling]) - (col_r + 0.5) * scaling
        y0 = -( np.array([i, i]) - (row_r + 0.5) * scaling)
        y1 = -(np.array([i + scaling, i + scaling]) - (row_r + 0.5) * scaling)
        plt.fill_between(x, y0, y1, color='grey')



def load_agent(resume_path):
    args = get_args_fetch()
    # create the ddpg_agent
    env = gym.make(args.env_name)
    test_env = gym.make(args.test)
    # rewrite args
    args.save = False
    args.resume = True
    args.resume_path = resume_path
    # get the environment parameters
    env_params = get_env_params(env)
    env_params['max_test_timesteps'] = test_env._max_episode_steps
    ddpg_trainer = ddpg_explore(args, env, env_params, test_env)
    return ddpg_trainer, args, env_params, env


def plot_graph(env, planner_policy, ddpg_trainer):
    start_time = time.time()
    observation = env.reset()
    obs = observation['observation']
    g = observation['desired_goal']
    ag = observation['achieved_goal']
    start_obs, goal = ddpg_trainer._preproc_inputs(obs, g)
    can_search = planner_policy._build_graph(start_obs, goal)
    print("build graph use:", time.time() - start_time)

    # use graph
    observation = env.reset()
    obs = observation['observation']
    g = observation['desired_goal']
    ag = observation['achieved_goal']
    start_obs, goal = ddpg_trainer._preproc_inputs(obs, g)
    can_search = planner_policy.use_graph(start_obs, goal)

    graph = planner_policy._g.copy()
    graph.remove_node('start')
    rb_vec = planner_policy.landmarks_tensor.cpu().numpy()[:, :2]
    # border = rb_vec[planner_policy.farthest_index]
    full_new_border = rb_vec[planner_policy.new_border]
    print("full_new_border", len(full_new_border))
    print("border valid prob", planner_policy.valid_prob)
    plot_real_graph(graph, rb_vec, ag, full_new_border)

def filter_duplicate_nodes(nodes):
    num = len(nodes)
    dist = np.zeros((num, num))
    for i in range(num):
        dist[i] = np.linalg.norm(nodes[i] - nodes, axis=1)
        min_2 = np.argpartition(dist[i], 2)[:2]
        for j in range(num):
            if dist[i][j] == 0 and i != j:
                print("duplicate !!!")


def plot_real_graph(g, rb_vec, start, border=None, plot_edges=False, plot_index=False):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    # plot_map()
    plt.scatter(rb_vec[g.nodes, 0], rb_vec[g.nodes, 1], label="node on graph")
    if plot_index:
        for node in g.nodes:
            ax.text(rb_vec[node, 0], rb_vec[node, 1], str(node), fontsize=16, color='red')

    edges_to_plot = g.edges
    edges_to_plot = np.array(list(edges_to_plot))
    print("Plotting {} nodes and {} edges".format(g.number_of_nodes(), len(edges_to_plot)))

    if plot_edges:
        for i, j in edges_to_plot:
            s_i = rb_vec[i]
            s_j = rb_vec[j]
            plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='k', alpha=0.5)

    if border is not None:
        plt.scatter(*border.T, c='red', label="border")

    # plot start point
    plt.plot([start[0]], [start[1]], c='k', marker="*", markersize=15, label="start")
    plt.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.15), ncol=3, fontsize=16)

    plt.show()
    # plt.savefig("./data/real_graph.png")


def rollout_with_search(agent, planner_policy, eval_tf_env, env_params, all_search=True):
    # @title Rollouts with Search. { vertical-output: true, run: "auto"}
    n_runs = 4
    for col_index in range(n_runs):
        title = 'no search' if col_index == 0 else 'search'
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        # plot_map()
        if all_search:
            use_search = True
        else:
            use_search = (col_index == 1)
        ts = eval_tf_env.reset()
        goal = ts['desired_goal']
        start = ag = ts['achieved_goal']
        obs = ts['observation']
        obs_vec = []
        done = False
        if use_search:
            start_obs, g = agent._preproc_inputs(obs, goal)
            can_search = planner_policy._build_graph(start_obs, g)
            print("can_search", can_search)
            rb_vec = planner_policy.landmarks_tensor.cpu().numpy()
        for _ in range(env_params['max_timesteps']):
            obs_vec.append(ag)
            if done:
                break
            with torch.no_grad():
                act_obs, act_g = agent._preproc_inputs(obs, goal)

                if use_search:
                    action = planner_policy(act_obs, act_g)
                else:
                    action = agent.test_policy(act_obs, act_g)

            observation_new, _, done, info = eval_tf_env.step(action)
            obs_new = observation_new['observation']
            ag_new = observation_new['achieved_goal']
            obs = obs_new
            ag = ag_new

        obs_vec = np.array(obs_vec)

        plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
        plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
                    color='red', s=200, label='start')
        plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
                    color='green', s=200, label='end')
        plt.scatter([goal[0]], [goal[1]], marker='*',
                    color='green', s=200, label='goal')

        plt.title(title, fontsize=24)
        if use_search and hasattr(planner_policy, '_waypoint_vec'):
            waypoint_vec = [start]
            for waypoint_index in planner_policy._waypoint_vec:
                waypoint_vec.append(rb_vec[waypoint_index])
            waypoint_vec.append(goal)
            waypoint_vec = np.array(waypoint_vec)
            print("way_point", waypoint_vec)

            for i in range(len(waypoint_vec)):
                ax.text(waypoint_vec[i,0], waypoint_vec[i,1], str(i), fontsize=16, color='red')

            plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'k-s', label='waypoint')
            plt.legend(loc='lower left', bbox_to_anchor=(-0.8, -0.15), ncol=4, fontsize=16)
        plt.show()
        plt.cla()




if __name__ == "__main__":
    resume_path = "../saved_models/FetchPush-v2_May03_08-37-25"
    # ddpg_trainer, args, env_params, env = load_agent(resume_path=resume_path)

    # extract mapping
    buffer, landmarks = extract_mapping(resume_path=resume_path)

    count_frontier(landmarks)


    n_landmarks = args.landmark
    # n_landmarks = 400
    # clip_v = args.clip_v
    clip_v = -60.
    # min_dist = args.min_dist
    min_dist = 0
    print("clip_v", clip_v)

    planner_policy = Frontier_explore(agent=ddpg_trainer, replay_buffer=buffer, goal_dim=env_params["goal"],
                             clip_v=clip_v)

    # planner_policy.k = 30
    # print("k", planner_policy.k)

    # plot_graph(env, planner_policy, ddpg_trainer)
    # rollout_with_search(ddpg_trainer, planner_policy, env, env_params)

