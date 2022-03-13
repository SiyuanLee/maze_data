import os
import sys

sys.path.append('../')
from datetime import datetime
from tensorboardX import SummaryWriter
from models.networks import *
from algos.replay_buffer import replay_buffer, replay_buffer_energy
from algos.her import her_sampler
from planner.goal_plan import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
from algos.sac.sac import SAC
from algos.sac.replay_memory import ReplayMemory
import gym


class double_sac_agent:
    def __init__(self, args, env, env_params, test_env):
        self.args = args
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.device = args.device
        self.resume = args.resume
        self.resume_epoch = args.resume_epoch
        self.not_train_low = False

        self.low_dim = env_params['obs']
        self.env_params['low_dim'] = self.low_dim
        self.hi_dim = env_params['obs']

        self.learn_goal_space = True
        self.whole_obs = False  # use whole observation space as subgoal space
        self.abs_range = abs_range = 50.  # absolute goal range
        self.feature_reg = 0.0  # feature l2 regularization

        if args.env_name[:5] == "Fetch":
            maze_low = self.env.env.initial_gripper_xpos[:2] - self.env.env.target_range
            maze_high = self.env.env.initial_gripper_xpos[:2] + self.env.env.target_range
            self.hi_act_space = gym.spaces.Box(low=maze_low, high=maze_high)
        else:
            self.hi_act_space = self.env.env.maze_space
        if self.learn_goal_space:
            self.hi_act_space = gym.spaces.Box(low=np.array([-abs_range, -abs_range]), high=np.array([abs_range, abs_range]))
        if self.whole_obs:
            vel_low = [-10.] * 4
            vel_high = [10.] * 4
            maze_low = np.concatenate((self.env.env.maze_low, np.array(vel_low)))
            maze_high = np.concatenate((self.env.env.maze_high, np.array(vel_high)))
            self.hi_act_space = gym.spaces.Box(low=maze_low, high=maze_high)

        # # params of learning phi
        # resume_phi = args.resume
        # self.not_update_phi = False
        # phi_path = args.resume_path

        resume_phi = True
        phi_path = 'saved_models/AntMaze1-v1_Jun01_19-26-19'
        self.not_update_phi = True

        self.early_stop = True  # after success rate converge, don't update low policy and feature
        self.success_log = []
        self.hi_sparse = (self.env.env.reward_type == "sparse")
        self.done_break = False

        if args.save:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.log_dir = 'runs/hier/ant/LowDone_D5_HiDense_' + current_time + "_DoneB_" + str(self.done_break) + \
                            "_C_" + str(args.c) + "_PhiL2_" + str(self.feature_reg) + \
                            "_LearnG_" + str(self.learn_goal_space)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name + "_" + current_time)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
        # init low-level network
        self.real_goal_dim = self.hi_act_space.shape[0]  # low-level goal space and high-level action space
        self.start_epoch = 0
        self.low_act_space = self.env.env.env.wrapped_env.action_space
        self.low_agent = SAC(self.low_dim + self.real_goal_dim, self.low_act_space, args, False)
        self.low_buffer = ReplayMemory(args.buffer_size)
        # init high-level agent
        self.hi_agent = SAC(self.hi_dim, self.hi_act_space, args, False)
        self.hi_buffer = ReplayMemory(args.buffer_size)

        self.c = args.c  # interval of high level action

        not_load_buffer, not_load_high = False, False
        if self.resume is True:
            self.start_epoch = self.resume_epoch
            if not not_load_high:
                self.hi_agent.policy.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/hi_actor_model.pt', map_location='cuda:1')[0])
                self.hi_agent.critic.load_state_dict(torch.load(self.args.resume_path + \
                                                               '/hi_critic_model.pt', map_location='cuda:1')[0])
            self.hi_agent.policy.load_state_dict(torch.load(self.args.resume_path + \
                                                             '/low_actor_model.pt', map_location='cuda:1')[0])
            self.hi_agent.critic.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/low_critic_model.pt', map_location='cuda:1')[0])
            if not not_load_buffer:
                self.hi_buffer = torch.load(self.args.resume_path + '/hi_buffer.pt', map_location='cuda:1')
                self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')

        if hasattr(self.env.env, 'env'):
            self.animate = self.env.env.env.visualize_goal
        else:
            self.animate = self.args.animate

        if args.contrastive_phi:
            self.representation = RepresentationNetwork(env_params, layer=3).to(args.device)
            self.representation_optim = torch.optim.Adam(self.representation.parameters(), lr=0.0001)
            if resume_phi is True:
                self.representation.load_state_dict(torch.load(phi_path + \
                                                               '/phi_model.pt', map_location='cuda:1')[0])

        print("learn goal space", self.learn_goal_space, " update phi", not self.not_update_phi)

    def learn(self):
        for epoch in range(self.start_epoch, self.args.n_epochs):
            last_hi_obs = None
            observation = self.env.reset()
            obs = observation['observation']

            for t in range(self.env_params['max_timesteps']):
                if t % self.c == 0:
                    # append high-level rollouts
                    if last_hi_obs is not None:
                        if done and self.done_break:
                            break
                        mask = float(not done)
                        self.hi_buffer.push(last_hi_obs, last_hi_a, last_hi_r, obs[:self.hi_dim], mask, epoch)
                    if epoch < self.args.start_epoch:
                        hi_action = self.hi_act_space.sample()
                    else:
                        hi_action = self.hi_agent.select_action(obs[:self.hi_dim])
                    last_hi_obs = obs[:self.hi_dim].copy()
                    last_hi_a = hi_action.copy()
                    last_hi_r = 0.
                    done = False
                    # update high-level policy
                    if len(self.hi_buffer) > self.args.batch_size:
                        self.update_hi(epoch)
                act_obs = np.concatenate((obs, hi_action))
                if self.not_train_low:
                    action = self.low_agent.select_action(act_obs, evaluate=True)
                else:
                    if epoch < self.args.low_start_epoch:
                        action = self.low_act_space.sample()
                    else:
                        action = self.low_agent.select_action(act_obs)
                # feed the actions into the environment
                observation_new, r, _, info = self.env.step(action)
                if done is False:
                    last_hi_r += r
                if info['is_success']:
                    done = True
                if self.animate:
                    self.env.render()
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                if self.learn_goal_space:
                    ag_new = self.representation(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                if self.whole_obs:
                    ag_new = obs_new.copy()
                # calculate low reward
                assert ag_new.shape == hi_action.shape
                low_r = - np.linalg.norm(ag_new - hi_action, axis=-1) / 100.0
                new_act_obs = np.concatenate((obs_new, hi_action))
                low_done = bool(low_r > -0.001)
                low_mask = float(not low_done)
                self.low_buffer.push(act_obs, action, low_r, new_act_obs, low_mask, epoch)
                # update low-level policy
                if len(self.low_buffer) > self.args.batch_size:
                    self.update_low(epoch)
                # re-assign the observation
                obs = obs_new
                # slowly update phi
                if epoch > 10 and not self.not_update_phi:
                    self.slow_update_phi(epoch)
            mask = float(not done)
            self.hi_buffer.push(last_hi_obs, last_hi_a, last_hi_r, obs[:self.hi_dim], mask, epoch)

            # start to do the evaluation
            if epoch % self.args.eval_interval == 0 and epoch != 0:
                hier_success_rate, discount_reward = self._eval_hier_agent()
                self.success_log.append(hier_success_rate)
                mean_success = np.mean(self.success_log[-5:])
                # stop updating phi and low
                if self.early_stop and mean_success >= 0.9:
                    print("early stop !!!")
                    self.not_update_phi = True
                    self.not_train_low = True
                print('[{}] epoch is: {}, eval hier success rate is: {:.3f}'.format(datetime.now(), epoch, hier_success_rate))

                if self.args.save:
                    print("log_dir: ", self.log_dir)
                    torch.save([self.hi_agent.policy.state_dict()], self.model_path + '/hi_actor_model.pt')
                    torch.save([self.hi_agent.critic.state_dict()], self.model_path + '/hi_critic_model.pt')
                    torch.save([self.low_agent.critic.state_dict()], self.model_path + '/low_critic_model.pt')
                    torch.save([self.low_agent.policy.state_dict()], self.model_path + '/low_actor_model.pt')
                    torch.save(self.hi_buffer, self.model_path + '/hi_buffer.pt')
                    torch.save(self.low_buffer, self.model_path + '/low_buffer.pt')
                    self.writer.add_scalar('Success_rate/hier_' + self.args.env_name + self.args.metric, hier_success_rate, epoch)
                    self.writer.add_scalar('Success_rate/hier_discount_reward', discount_reward, epoch)
                    if self.args.contrastive_phi:
                        torch.save([self.representation.state_dict()], self.model_path + '/phi_model.pt')

    def update_hi(self, epoch):
        critic_1_loss, critic_2_loss, policy_loss, _, _ = self.hi_agent.update_parameters(self.hi_buffer,
                                                                                          self.args.batch_size,
                                                                                          self.env_params,
                                                                                          self.hi_sparse)
        if self.args.save:
            self.writer.add_scalar('Loss/hi_critic_1', critic_1_loss, epoch)
            self.writer.add_scalar('Loss/hi_critic_2', critic_2_loss, epoch)
            self.writer.add_scalar('Loss/hi_policy', policy_loss, epoch)

    def update_low(self, epoch):
        critic_1_loss, critic_2_loss, policy_loss, _, _ = self.low_agent.update_parameters(self.low_buffer,
                                                                                            self.args.batch_size,
                                                                                            None,
                                                                                            False)
        if self.args.save:
            self.writer.add_scalar('Loss/low_critic_1', critic_1_loss, epoch)
            self.writer.add_scalar('Loss/low_critic_2', critic_2_loss, epoch)
            self.writer.add_scalar('Loss/low_policy', policy_loss, epoch)

    def _eval_hier_agent(self, n_test_rollouts=100):
        total_success_rate = []
        if not self.args.eval:
            n_test_rollouts = self.args.n_test_rollouts
        discount_reward = np.zeros(n_test_rollouts)
        for roll in range(n_test_rollouts):
            per_success_rate = []
            observation = self.test_env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for num in range(self.env_params['max_test_timesteps']):
                with torch.no_grad():
                    if num % self.c == 0:
                        hi_action = self.hi_agent.select_action(obs[:self.hi_dim], evaluate=True)
                    act_obs = np.concatenate((obs, hi_action))
                    action = self.low_agent.select_action(act_obs, evaluate=True)
                observation_new, rew, done, info = self.test_env.step(action)
                if self.animate:
                    self.test_env.render()
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                if done:
                    per_success_rate.append(info['is_success'])
                    if bool(info['is_success']):
                        discount_reward[roll] = 1 - 1. / self.env_params['max_test_timesteps'] * num
                    break
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        global_reward = np.mean(discount_reward)
        if self.args.eval:
            print("hier success rate", global_success_rate, global_reward)
        return global_success_rate, global_reward

    def update_phi(self, epoch, not_align=True):
        # use all data
        train_data = self.new_collect()
        shuffle_idx = np.random.permutation(train_data.shape[1])
        train_data = train_data[:, shuffle_idx]
        n_batches = int(train_data.shape[1] / self.args.batch_size)
        for n_batch in range(n_batches):
            from_idx, to_idx = n_batch * self.args.batch_size, (n_batch + 1) * self.args.batch_size
            sample_data = train_data[:, from_idx:to_idx]
            sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
            obs, obs_next = self.representation(sample_data[0]), self.representation(sample_data[1])
            min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
            if not_align:
                hi_obs, hi_obs_next = self.representation(sample_data[2]), self.representation(sample_data[3])
                max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
            else:
                neg_obs = self.representation(sample_data[2])
                max_dist = torch.clamp(1 - (obs - neg_obs).pow(2).mean(dim=1), min=0.)
            representation_loss = (min_dist + max_dist).mean()
            # add l2 regularization
            representation_loss += self.feature_reg * (obs / self.abs_range).pow(2).mean()
            self.representation_optim.zero_grad()
            representation_loss.backward()
            self.representation_optim.step()
            if self.args.save:
                self.writer.add_scalar('Loss/phi_loss' + self.args.metric, representation_loss, epoch)

    def new_collect(self):
        all_obs = self.low_buffer.buffers['obs'][:self.low_buffer.current_size]
        obs = all_obs[:, :-1, :]
        obs_next = all_obs[:, 1:, :]
        obs, obs_next = obs.reshape(-1, self.env_params['obs']), obs_next.reshape(-1, self.env_params['obs'])
        len_obs = obs.shape[0]

        hi_obs, hi_obs_next = self.hi_buffer.get_obs()

        tile_num = int(len_obs / self.c)
        hi_obs = np.tile(hi_obs[-tile_num:], (self.c, 1))
        hi_obs_next = np.tile(hi_obs_next[-tile_num:], (self.c, 1))

        train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
        return train_data

    def visualize_representation(self, epoch):
        obs = self.low_buffer.get_all_data()['obs'].copy()
        obs = obs.reshape(-1, obs.shape[-1])
        select = np.random.choice(obs.shape[0], 1000, replace=False)
        obs = obs[select]
        start_time = time.time()
        index1 = np.where((obs[:, 0] < 4) & (obs[:, 1] < 4))
        index2 = np.where((obs[:, 0] < 4) & (obs[:, 1] > 4))
        index3 = np.where((obs[:, 0] > 4) & (obs[:, 1] < 4))
        index4 = np.where((obs[:, 0] > 4) & (obs[:, 1] > 4))
        index_lst = [index1, index2, index3, index4]

        obs_tensor = torch.Tensor(obs).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        rep = []
        for index in index_lst:
            rep.append(features[index])
        print("inference use ", time.time() - start_time, " seconds")

        obs_list = []
        for index in index_lst:
            obs_list.append(obs[index])

        start_time = time.time()
        tsne_list = []
        res_tsne = TSNE(n_components=2).fit_transform(obs)
        for index in index_lst:
            tsne_list.append(res_tsne[index])
        print("tsne use ", time.time() - start_time, " seconds")

        start_time = time.time()
        self.plot_fig(obs_list, 'obs', epoch)
        self.plot_fig(rep, 'margin_feature', epoch)
        self.plot_fig(tsne_list, 'tsne_feature', epoch)
        print("plot fig use ", time.time() - start_time, " seconds")

    def plot_fig(self, rep, name, epoch):
        fig = plt.figure()
        axes = fig.add_subplot(111)
        rep1, rep2, rep3, rep4 = rep
        def scatter_rep(rep1, c, marker):
            if rep1.shape[0] > 0:
                l1 = axes.scatter(rep1[:, 0], rep1[:, 1], c=c, marker=marker)
            else:
                l1 = axes.scatter([], [], c=c, marker=marker)
            return l1
        l1 = scatter_rep(rep1, c='y', marker='s')
        l2 = scatter_rep(rep2, c='r', marker='o')
        l3 = scatter_rep(rep3, c='b', marker='1')
        l4 = scatter_rep(rep4, c='g', marker='2')

        plt.xlabel('x')
        plt.ylabel('y')
        axes.legend((l1, l2, l3, l4), ('space1', 'space2', 'space3', 'space4'))
        plt.savefig('fig/' + name + str(epoch) + '.png')
        plt.close()


    def vis_hier_policy(self, epoch=None):
        obs_vec = []
        hi_action_vec= []
        per_success_rate = []
        observation = self.test_env.reset()
        obs = observation['observation']
        obs_vec.append(obs)
        g = observation['desired_goal']
        for num in range(self.env_params['max_test_timesteps']):
            with torch.no_grad():
                if num % self.c == 0:
                    hi_action = self.hi_agent.select_action(obs[:self.hi_dim], evaluate=True)
                    hi_action_vec.append(hi_action)
                act_obs = np.concatenate((obs, hi_action))
                action = self.low_agent.select_action(act_obs, evaluate=True)
            observation_new, rew, done, info = self.test_env.step(action)
            if self.animate:
                self.test_env.render()
            obs = observation_new['observation']
            obs_vec.append(obs)
            if done:
                per_success_rate.append(info['is_success'])
                break

        plt.figure(figsize=(12, 6))
        obs_vec = np.array(obs_vec)
        self.plot_rollout(obs_vec, "xy_rollout", 121, goal=g)

        obs_tensor = torch.Tensor(obs_vec).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        rest = (self.env_params['obs'] - self.real_goal_dim) * [0.]
        g = np.concatenate((g, np.array(rest)))
        g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
        feature_goal = self.representation(g).detach().cpu().numpy()[0]
        hi_action_vec = np.array(hi_action_vec)
        self.plot_rollout(features, "hier_rollout", 122, feature_goal, hi_action_vec)
        file_name = 'fig/rollout' + str(epoch) + '.png'
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

    def plot_rollout(self, obs_vec, name, num, goal=None, hi_action_vec=None):
        plt.subplot(num)
        cm = plt.cm.get_cmap('RdYlBu')
        num = np.arange(obs_vec.shape[0])
        plt.scatter(obs_vec[:, 0], obs_vec[:, 1], c=num, cmap=cm)
        plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
                    color='green', s=200, label='start')
        plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
                    color='red', s=200, label='end')
        if goal is not None:
            plt.scatter([goal[0]], [goal[1]], marker='*',
                        color='green', s=200, label='goal')
        if hi_action_vec is not None:
            hi_action_num = hi_action_vec.shape[0]
            for hi_action_index in range(hi_action_num):
                pos = obs_vec[hi_action_index * self.c]
                hi_action = hi_action_vec[hi_action_index]
                plt.arrow(pos[0], pos[1], hi_action[0], hi_action[1], head_width=0.3)
        plt.legend(loc=2,  bbox_to_anchor=(1.05, 1.0), fontsize=14, borderaxespad=0.)
        plt.title(name, fontsize=24)

    def slow_update_phi(self, epoch):
        sample_data = self.slow_collect()
        sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
        obs, obs_next = self.representation(sample_data[0]), self.representation(sample_data[1])
        min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
        hi_obs, hi_obs_next = self.representation(sample_data[2]), self.representation(sample_data[3])
        max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
        representation_loss = (min_dist + max_dist).mean()
        self.representation_optim.zero_grad()
        representation_loss.backward()
        self.representation_optim.step()
        if self.args.save:
            self.writer.add_scalar('Loss/phi_loss' + self.args.metric, representation_loss, epoch)

    def slow_collect(self, batch_size=100):
        obs, _, _, obs_next, _ = self.low_buffer.sample(batch_size)
        obs, obs_next = obs[:, :self.env_params['obs']], obs_next[:, :self.env_params['obs']]

        hi_obs, _, _, hi_obs_next, _ = self.hi_buffer.sample(batch_size)

        train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
        return train_data


