import torch
import numpy as np
from .sample import farthest_point_sample
from torch.distributions import Categorical
import networkx as nx
import time


class Planner:
    def __init__(self, agent, replay_buffer, goal_dim, min_dist, n_landmark=200, clip_v=-4,
                 eval=False, k=50):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.n_landmark = n_landmark
        self.clip_v = -clip_v

        self.trust_region = -clip_v
        self.k = k
        self.fake_goal_num = 1e-10
        self.reach_fake_num = 0
        self.train = not eval
        self.percept_thres = self.agent.args.percept_thres
        self.follow_thres = self.agent.args.follow_thres
        self.goal_dim = goal_dim
        self.follow_path = False
        self.valid_prob = 0.
        self.min_dist = torch.tensor(min_dist).to(self.agent.device)
        self.max_try = 10000
        self._g = nx.DiGraph()
        print("min_dist", self.min_dist)
        print("goal_dim in explore", goal_dim)

        if self.train:
            self.policy = self.agent.explore_policy
        else:
            self.policy = self.agent.test_policy

    def _build_graph(self, first_state, goal, balance=True):
        self._g = nx.DiGraph()
        with torch.no_grad():
            dist_to_goal = self.agent.pairwise_value(first_state,
                                                     goal)
            # print("dist to goal", dist_to_goal)
        if dist_to_goal < self.trust_region:
            # do not need search
            self.follow_path = False
            return False, None

        state = self.replay_buffer.get_all_data()['obs']

        # get a index array
        episode_num, ep_len, _ = state.shape
        index_array = np.zeros((episode_num * ep_len, 2), dtype=int)
        timestep = np.array(list(range(ep_len)) * episode_num)
        index_array[:, 1] = timestep
        episode = np.arange(episode_num).repeat(ep_len)
        index_array[:, 0] = episode

        state = state.reshape(-1, state.shape[2])
        landmarks = state[:, :self.goal_dim]

        if balance:
            rb_vec, index_lst, dists = self.get_samples(landmarks, state)
        else:
            index_lst = np.random.choice(len(landmarks), self.n_landmark)
            rb_vec = landmarks[index_lst]

        state_graph = state[index_lst]
        select_index = index_array[index_lst]
        self.state_tensor = torch.Tensor(state_graph).to(self.agent.device)
        self.landmarks_tensor = torch.Tensor(rb_vec).to(self.agent.device)

        can_connect_start = self.connect_start(first_state)
        if not can_connect_start:
            self.follow_path = False
            return False, None

        if not balance:
            dists = self.pairwise_dists(self.state_tensor, self.landmarks_tensor)
        dists = torch.max(dists, dists * 0)
        # build graph
        out_count = self.connect_nodes(dists)
        # new frontier
        new_border = np.argpartition(out_count, self.k)[:self.k]
        self.new_border = [i for i in new_border if i in self._g.nodes]
        border_index = select_index[self.new_border]
        self._waypoint_vec = self._get_path(first_state, goal)
        if self._waypoint_vec is None:
            # identify new_border is valid or not
            self.get_border()
            if not bool(self.farthest_index):
                self.follow_path = False
                return False, border_index
            else:
                # sample a fake goal
                self.sample_fake_goal()
        self.fake_goal_num += 1.
        self._waypoint_counter = 0
        self.fake_goal = self.landmarks_tensor[self._waypoint_vec[-1]]
        self.follow_path = True
        return True, border_index

    def use_graph(self, first_state, goal):
        with torch.no_grad():
            dist_to_goal = self.agent.pairwise_value(first_state,
                                                     goal)
            # print("dist to goal", dist_to_goal)
        if dist_to_goal < self.trust_region:
            # do not need search
            self.follow_path = False
            return False

        self._g.remove_node('start')
        can_connect_start = self.connect_start(first_state)
        if not can_connect_start:
            self.follow_path = False
            return False

        self._waypoint_vec = self._get_path(first_state, goal)
        if self._waypoint_vec is None:
            # identify new_border is valid or not
            self.get_border()
            if not bool(self.farthest_index):
                self.follow_path = False
                return False
            else:
                # sample a fake goal
                self.sample_fake_goal()
        self.fake_goal_num += 1.
        self._waypoint_counter = 0
        self.fake_goal = self.landmarks_tensor[self._waypoint_vec[-1]]
        self.follow_path = True
        return True

    def _get_path(self, obs, goal):
        with torch.no_grad():
            goal = goal.expand(self.n_landmark, *goal.shape[1:])
            rb_to_goal = self.agent.pairwise_value(self.state_tensor,
                                                   goal)
        if torch.min(rb_to_goal) >= self.clip_v:
            return None
        else:
            g2 = self._g.copy()
            for i, dist_to_goal in enumerate(rb_to_goal):
                if dist_to_goal < self.clip_v:
                    g2.add_edge(i, 'goal', weight=dist_to_goal)
            try:
                # try to get a path from start to goal
                path = nx.shortest_path(g2, 'start', 'goal')
                waypoint_vec = list(path)[1:-1]
                return waypoint_vec
            except:
                return None

    # sample a fake goal from a candidate set
    def sample_fake_goal(self):
        index = np.random.randint(0, len(self.farthest_index))
        self._waypoint_vec = self.farthest_paths[index]

    def pairwise_dists(self, states, landmarks):
        with torch.no_grad():
            dists = []
            obs = states
            for i in landmarks:
                goal = i[None, :].expand(len(states), *i.shape)
                dists.append(self.agent.pairwise_value(obs, goal))
        return torch.stack(dists, dim=1)

    def __call__(self, obs, goal=None):
        if isinstance(obs, np.ndarray):
            obs = torch.Tensor(obs).to(self.agent.device)

        if isinstance(goal, np.ndarray):
            goal = torch.Tensor(goal).to(self.agent.device)

        if not self.follow_path:
            return self.policy(obs, goal)
        else:
            real_dist_to_fakegoal = self.get_real_dist(obs, self.fake_goal)
            if real_dist_to_fakegoal < self.follow_thres:
                self.follow_path = False
                self.reach_fake_num += 1.
                return self.policy(obs, goal)
            else:
                # try to follow path
                waypoint = self.landmarks_tensor[self._waypoint_vec[self._waypoint_counter]]
                real_dist_to_waypoint = self.get_real_dist(obs, waypoint)
                if real_dist_to_waypoint < self.follow_thres:
                    self._waypoint_counter = min(self._waypoint_counter + 1,
                                                 len(self._waypoint_vec) - 1)
                    waypoint = self.landmarks_tensor[self._waypoint_vec[self._waypoint_counter]]
                return self.agent.test_policy(obs, waypoint[None])

    def get_vec_dist(self, point, vec):
        real_dist = np.linalg.norm(vec - point, axis=1)
        return real_dist

    def get_samples(self, landmarks, state):
        if self.agent.args.env_name[:3] == "Ant":
            height_arr = state[:, 2]
            valid_indexs = np.where((height_arr > 0.3) & (height_arr < 1.0))[0]
        else:
            valid_indexs = np.arange(len(landmarks))
        np.random.shuffle(valid_indexs)
        new_rb_vec = []
        index_lst = []
        random_idx = valid_indexs[0]

        new_rb_vec.append(landmarks[random_idx])
        index_lst.append(random_idx)
        # convert to tensor
        state_vec = state[index_lst]
        self.state_for_dist = torch.Tensor(state_vec).to(self.agent.device)
        self.goal_tensor = self.state_for_dist[:, :self.goal_dim]

        dist_matrix = torch.ones((self.n_landmark, self.n_landmark)).to(self.agent.device)

        count = 1
        for random_idx in valid_indexs[1:self.max_try]:
            if count == self.n_landmark:
                break
            # perception similarity
            real_dist = self.get_vec_dist(landmarks[random_idx], np.array(new_rb_vec))
            if min(real_dist) > self.percept_thres:
                new = state[random_idx]
                act_dist, new_to_rb, rb_to_new = self.actionable_dist(new)
                if act_dist > self.min_dist:
                    new_rb_vec.append(landmarks[random_idx])
                    index_lst.append(random_idx)
                    # keep dist matrix
                    dist_matrix[count, :count] = new_to_rb
                    dist_matrix[:count, count] = rb_to_new
                    count += 1
                    # convert to tensor
                    state_vec = state[index_lst]
                    self.state_for_dist = torch.Tensor(state_vec).to(self.agent.device)
                    self.goal_tensor = self.state_for_dist[:, :self.goal_dim]
                    # print("count:", count)

        new_rb_vec = np.array(new_rb_vec)
        index_lst = np.array(index_lst)
        print("random nodes number:", self.n_landmark - count)
        if count != self.n_landmark:
            random_idx = np.random.choice(valid_indexs[self.max_try:], self.n_landmark - count, replace=False)
            new_rb_vec = np.concatenate((new_rb_vec, landmarks[random_idx]), axis=0)
            index_lst = np.concatenate((index_lst, random_idx))

            dist_matrix = self.get_rest_dists(dist_matrix, random_idx, index_lst, state)

        return new_rb_vec, index_lst, dist_matrix

    # get border of the graph, top k farthest
    def get_border(self):
        path_record = [None] * self.n_landmark
        valid_i = []
        for i in self.new_border:
            try:
                path = nx.shortest_path(self._g, 'start', i)
                path_record[i] = path
                valid_i.append(i)
            except:
                pass
        if valid_i != []:
            self.farthest_index = valid_i
            self.valid_prob = len(self.farthest_index) / float(len(self.new_border))
            self.farthest_paths = [path_record[i][1:] for i in self.farthest_index]
        else:
            self.farthest_index = []

    def get_real_dist(self, obs, waypoint):
        obs = obs[0][:self.agent.real_goal_dim]
        real_dist = (obs - waypoint[:self.agent.real_goal_dim]).norm(2)
        return real_dist

    def actionable_dist(self, new):
        with torch.no_grad():
            new = torch.Tensor(new).to(self.agent.device)[None]
            new_goal = new[:, :self.goal_dim]

            new_goal = new_goal.expand(len(self.state_for_dist), *new_goal.shape[1:])
            rb_to_new = self.agent.pairwise_value(self.state_for_dist,
                                                  new_goal)

            new = new.expand(len(self.state_for_dist), *new.shape[1:])
            new_to_rb = self.agent.pairwise_value(new,
                                                  self.goal_tensor)
            dist = (new_to_rb + rb_to_new) / 2
            return torch.min(dist), new_to_rb, rb_to_new

    def connect_start(self, first_state):
        assert len(first_state) == 1
        with torch.no_grad():
            first_state = first_state.expand(self.n_landmark, *first_state.shape[1:])
            start_to_rb = self.agent.pairwise_value(first_state,
                                                    self.landmarks_tensor)
        if torch.min(start_to_rb) >= self.trust_region:
            return False
        else:
            for i, dist_from_start in enumerate(start_to_rb):
                if dist_from_start < self.trust_region:
                    self._g.add_edge('start', i, weight=dist_from_start)
            return True

    def get_rest_dists(self, dist_matrixs, new_lst, all_lst, state):
        all_state = state[all_lst]
        new_state = state[new_lst]

        all_state = torch.Tensor(all_state).to(self.agent.device)
        all_goal = all_state[:, :self.goal_dim]

        new_state = torch.Tensor(new_state).to(self.agent.device)
        new_goal = new_state[:, :self.goal_dim]

        cols = self.pairwise_dists(all_state, new_goal)
        rows = self.pairwise_dists(new_state, all_goal)

        len_new = len(new_lst)
        dist_matrixs[:, -len_new:] = cols
        dist_matrixs[-len_new:, :] = rows
        return dist_matrixs

    def connect_nodes(self, dists):
        dists = dists.cpu().numpy()
        out_count = np.zeros(self.n_landmark, dtype=np.int32)
        for i in range(self.n_landmark):
            for j in range(self.n_landmark):
                if i != j:
                    length = dists[i, j]
                    if length < self.clip_v:
                        self._g.add_edge(i, j, weight=length)
                        out_count[i] += 1
        return out_count
