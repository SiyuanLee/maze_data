import torch
import numpy as np
from .simhash import HashingBonusEvaluator


class Frontier_explore:
    def __init__(self, agent, replay_buffer, goal_dim, n_frontier=50, clip_v=-4,
                 hash_k_dim=128):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.n_frontier = n_frontier
        self.trust_region = -clip_v
        self.fake_goal_num = 1e-10
        self.reach_fake_num = 0
        self.follow_thres = self.agent.args.follow_thres
        self.follow_path = False
        self.hash = HashingBonusEvaluator(hash_k_dim, goal_dim)
        self.policy = self.agent.explore_policy
        print("goal_dim in explore", goal_dim)

    def select_frontier(self, first_state, goal):
        with torch.no_grad():
            dist_to_goal = self.agent.pairwise_value(first_state,
                                                     goal)
        if dist_to_goal < self.trust_region:
            # do not need search
            self.follow_path = False
            return False

        ag = self.replay_buffer.get_all_data()['ag'].copy()
        ag = ag.reshape(-1, ag.shape[2])
        counts = self.hash.predict(ag)

        idx = self.sample_frontier(counts, n_frontier=self.n_frontier)
        self.fake_goal_num += 1.
        self.fake_goal = torch.Tensor(ag[idx][None]).to(self.agent.device)

        self.follow_path = True
        return True


    def __call__(self, obs, goal=None, t=0):
        if isinstance(obs, np.ndarray):
            obs = torch.Tensor(obs).to(self.agent.device)
        if isinstance(goal, np.ndarray):
            goal = torch.Tensor(goal).to(self.agent.device)

        if not self.follow_path or t > self.agent.env_params['max_timesteps'] / 2:
            return self.policy(obs, goal)
        else:
            real_dist_to_fakegoal = self.get_real_dist(obs, self.fake_goal)
            if real_dist_to_fakegoal < self.follow_thres:
                self.follow_path = False
                self.reach_fake_num += 1.
                return self.policy(obs, goal)
            else:
                return self.agent.test_policy(obs, self.fake_goal)

    def sample_frontier(self, counts, n_frontier):
        idxs = np.argpartition(counts, n_frontier)[:n_frontier]
        frontier_index = np.random.randint(0, n_frontier)
        idx = idxs[frontier_index]
        return idx

    def get_real_dist(self, obs, waypoint):
        obs = obs[0][:self.agent.real_goal_dim]
        real_dist = (obs - waypoint[0][:self.agent.real_goal_dim]).norm(2)
        return real_dist

