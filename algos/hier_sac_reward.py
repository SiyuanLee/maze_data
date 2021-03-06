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
import random
from algos.sac.sac import SAC
from algos.sac.replay_memory import ReplayMemory, Array_ReplayMemory
import gym
import pickle
from planner.simhash import HashingBonusEvaluator
from planner.grid_hash import GridHashing
import imageio
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_color_codes()
import matplotlib
import matplotlib as mpl

# mpl.style.use('seaborn')

SUBGOAL_RANGE = 200.0


class hier_sac_reward_agent:
    def __init__(self, args, env, env_params, test_env, test_env1=None, test_env2=None):
        self.args = args
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.device = args.device
        self.resume = args.resume
        self.resume_epoch = args.resume_epoch
        self.not_train_low = False
        self.learn_hi = True
        self.test_env1 = test_env1
        self.test_env2 = test_env2
        self.old_sample = args.old_sample
        with open('fig/trajectory/' + 'good_maze_rollout.pkl', 'rb') as output:
            self.trajectory = pickle.load(output)

        self.low_dim = env_params['obs']
        self.env_params['low_dim'] = self.low_dim
        self.hi_dim = env_params['obs']
        print("hi_dim", self.hi_dim)

        self.learn_goal_space = True
        self.not_update_phi = False
        self.whole_obs = False  # use whole observation space as subgoal space
        self.abs_range = abs_range = args.abs_range  # absolute goal range
        self.feature_reg = 0.0  # feature l2 regularization
        print("abs_range", abs_range)

        if args.env_name[:5] == "Fetch":
            maze_low = self.env.env.initial_gripper_xpos[:2] - self.env.env.target_range
            maze_high = self.env.env.initial_gripper_xpos[:2] + self.env.env.target_range
            self.hi_act_space = gym.spaces.Box(low=maze_low, high=maze_high)
        else:
            if args.env_name != "NChain-v1":
                self.hi_act_space = self.env.env.maze_space
            else:
                self.hi_act_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]))
        if self.learn_goal_space:
            if args.env_name == "NChain-v1":
                self.hi_act_space = gym.spaces.Box(low=np.array([-abs_range]), high=np.array([abs_range]))
            else:
                self.hi_act_space = gym.spaces.Box(low=np.array([-abs_range, -abs_range]), high=np.array([abs_range, abs_range]))
        if self.whole_obs:
            vel_low = [-10.] * 4
            vel_high = [10.] * 4
            maze_low = np.concatenate((self.env.env.maze_low, np.array(vel_low)))
            maze_high = np.concatenate((self.env.env.maze_high, np.array(vel_high)))
            self.hi_act_space = gym.spaces.Box(low=maze_low, high=maze_high)


        dense_low = True
        self.low_use_clip = not dense_low  # only sparse reward use clip
        if args.replay_strategy == "future":
            self.low_forward = True
            assert self.low_use_clip is True
        else:
            self.low_forward = False
            assert self.low_use_clip is False
        self.hi_sparse = (self.env.env.reward_type == "sparse")

        # # params of learning phi
        resume_phi = args.resume
        phi_path = args.resume_path

        self.save_fig = False
        self.save_model = True
        self.start_update_phi = args.start_update_phi
        self.phi_interval = 100
        self.early_stop = args.early_stop  # after success rate converge, don't update low policy and feature
        if args.early_stop:
            if args.env_name in ['AntPush-v1', 'AntFall-v1']:
                self.early_stop_thres = 3500
            elif args.env_name in ["PointMaze1-v1"]:
                self.early_stop_thres = 2000
            elif args.env_name == "AntMaze1-v1":
                self.early_stop_thres = args.n_epochs
            else:
                self.early_stop_thres = 20000
        else:
            self.early_stop_thres = args.n_epochs
        print("early_stop_threshold", self.early_stop_thres)
        self.success_log = []

        self.scaling = scaling = self.env.env.env.MAZE_SIZE_SCALING
        print("scaling", scaling)
        print("reward type:", self.env.env.reward_type)

        self.count_latent = True
        self.usual_update_hash = False
        grid_scale = 0.5
        if self.count_latent:
            self.hash = GridHashing(grid_scale, obs_processed_flat_dim=2)
            self.xy_hash = GridHashing(grid_scale * 0.2, obs_processed_flat_dim=2)
        self.count_xy_record = []
        self.subgoal_record = []
        self.valid_times = 0
        self.start_explore = self.phi_interval
        self.intrinsic_coeff = 1

        self.count_obs = False
        if self.count_obs:
            self.hash = HashingBonusEvaluator(512, env_params['obs'])

        self.high_correct = False
        self.k = args.c
        self.delta_k = 0
        self.prediction_coeff = 0.0
        tanh_output = False
        self.use_prob = False
        print("prediction_coeff", self.prediction_coeff)

        if args.save:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.log_dir = 'runs/hier/' + str(args.env_name) + '/100_5w_Manual_False_' + current_time + \
                            "_C_" + str(args.c) + "_Image_" + str(args.image) + "_Early_" + str(self.early_stop_thres)  + \
                            "_Seed_" + str(args.seed) + "_Abs_" + str(args.abs_range) + '_Usual_' + str(self.usual_update_hash) + \
                            "_NoPhi_" + str(self.not_update_phi) + "_Start_" + str(self.start_explore) + \
                            "_LowCoeff_" + str(args.low_reward_coeff) + "_Old_" + str(args.old_sample) + "_Weight_" + str(args.weight_decay) + \
                             self.env.env.reward_type.upper() + '_Count_' + str(grid_scale) + str(self.count_latent) + "_Intrinsic_" + str(self.intrinsic_coeff)

            self.writer = SummaryWriter(log_dir=self.log_dir)
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name + "_" + current_time)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            self.fig_path = os.path.join(self.model_path, "figs")
            if not os.path.exists(self.fig_path):
                os.mkdir(self.fig_path)
                print("fig path", self.fig_path)
        # init low-level network
        self.real_goal_dim = self.hi_act_space.shape[0]  # low-level goal space and high-level action space
        self.init_network()
        # init high-level agent
        self.hi_agent = SAC(self.hi_dim + env_params['goal'], self.hi_act_space, args, False, env_params['goal'],
                            args.gradient_flow_value, args.abs_range, tanh_output)
        self.env_params['real_goal_dim'] = self.real_goal_dim
        self.hi_buffer = ReplayMemory(args.buffer_size)

        # her sampler
        self.c = self.args.c  # interval of high level action
        self.low_her_module = her_sampler(args.replay_strategy, args.replay_k, args.distance, args.future_step,
                                          dense_reward=dense_low, direction_reward=False, low_reward_coeff=args.low_reward_coeff)
        if args.env_name[:5] == "Fetch":
            self.low_buffer = replay_buffer_energy(self.env_params, self.args.buffer_size,
                                               self.low_her_module.sample_her_energy, args.env_name)
        else:
            self.low_buffer = replay_buffer(self.env_params, self.args.buffer_size, self.low_her_module.sample_her_transitions)

        not_load_buffer, not_load_high = True, False
        if self.resume is True:
            self.start_epoch = self.resume_epoch
            if not not_load_high:
                print("load high !!!")
                self.hi_agent.policy.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/hi_actor_model.pt', map_location='cuda:1')[0])
                self.hi_agent.critic.load_state_dict(torch.load(self.args.resume_path + \
                                                               '/hi_critic_model.pt', map_location='cuda:1')[0])

            # print("not load low !!!")
            print("load low !!!")
            self.low_actor_network.load_state_dict(torch.load(self.args.resume_path + \
                                                             '/low_actor_model.pt', map_location='cuda:1')[0])
            self.low_critic_network.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/low_critic_model.pt', map_location='cuda:1')[0])

            if not not_load_buffer:
                # self.hi_buffer = torch.load(self.args.resume_path + '/hi_buffer.pt', map_location='cuda:1')
                self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')

        # sync target network of low-level
        self.sync_target()

        if hasattr(self.env.env, 'env'):
            self.animate = self.env.env.env.visualize_goal
        else:
            self.animate = self.args.animate
        self.distance_threshold = self.args.distance

        if not (args.gradient_flow or args.use_prediction or args.gradient_flow_value):
            self.representation = RepresentationNetwork(env_params, 3, self.abs_range, self.real_goal_dim).to(args.device)
            if args.use_target:
                self.target_phi = RepresentationNetwork(env_params, 3, self.abs_range, 2).to(args.device)
                # load the weights into the target networks
                self.target_phi.load_state_dict(self.representation.state_dict())
            self.representation_optim = torch.optim.Adam(self.representation.parameters(), lr=0.0001)
            if resume_phi is True:
                print("load phi from: ", phi_path)
                self.representation.load_state_dict(torch.load(phi_path + \
                                                               '/phi_model_6000.pt', map_location='cuda:1')[0])
        elif args.use_prediction:
            self.representation = DynamicsNetwork(env_params, self.abs_range, 2, tanh_output=tanh_output, use_prob=self.use_prob, device=args.device).to(args.device)
            self.representation_optim = torch.optim.Adam(self.representation.parameters(), lr=0.0001)
            if resume_phi is True:
                print("load phi from: ", phi_path)
                self.representation.load_state_dict(torch.load(phi_path + \
                                                               '/phi_model_4000.pt', map_location='cuda:1')[0])



        print("learn goal space", self.learn_goal_space, " update phi", not self.not_update_phi)
        self.train_success = 0.
        self.count_prob = 1.
        self.furthest_task = 0.

    def adjust_lr_actor(self, epoch):
        lr_actor = self.args.lr_actor * (0.5 ** (epoch // self.args.lr_decay_actor))
        for param_group in self.low_actor_optim.param_groups:
            param_group['lr'] = lr_actor

    def adjust_lr_critic(self, epoch):
        lr_critic = self.args.lr_critic * (0.5 ** (epoch // self.args.lr_decay_critic))
        for param_group in self.low_critic_optim.param_groups:
            param_group['lr'] = lr_critic

    def learn(self):
        for epoch in range(self.start_epoch, self.args.n_epochs):
            if epoch > 0 and epoch % self.args.lr_decay_actor == 0:
                self.adjust_lr_actor(epoch)
            if epoch > 0 and epoch % self.args.lr_decay_critic == 0:
                self.adjust_lr_critic(epoch)

            ep_obs, ep_ag, ep_g, ep_actions, ep_ag_record = [], [], [], [], []
            last_hi_obs = None
            success = 0
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal'][:self.real_goal_dim]
            ag_record = observation['achieved_goal']
            g = observation['desired_goal']
            # identify furthest task
            if g[1] >= 8:
                self.furthest_task += 1
                is_furthest_task = True
            else:
                is_furthest_task = False
            if self.learn_goal_space:
                if self.args.gradient_flow:
                    if self.args.use_target:
                        ag = self.hi_agent.policy_target.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()
                    else:
                        ag = self.hi_agent.policy.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()
                elif self.args.gradient_flow_value:
                    ag = self.hi_agent.critic.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                elif self.args.use_prediction:
                    ag = self.representation.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                else:
                    if self.args.use_target:
                        ag = self.target_phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                    else:
                        ag = self.representation(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
            if self.whole_obs:
                ag = obs.copy()

            for t in range(self.env_params['max_timesteps']):
                act_obs, act_g = self._preproc_inputs(obs, g)
                if t % self.c == 0:
                    hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
                    # append high-level rollouts
                    if last_hi_obs is not None:
                        mask = float(not done)
                        if self.high_correct:
                            last_hi_a = ag
                        self.hi_buffer.push(last_hi_obs, last_hi_a, last_hi_r, hi_act_obs, mask, epoch)
                    random_num = random.random()
                    valid_count = bool(random_num < self.count_prob)
                    # select subgoal by sac
                    if epoch < self.args.start_epoch:
                        hi_action = self.hi_act_space.sample()
                    else:
                        hi_action = self.hi_agent.select_action(hi_act_obs)
                    last_hi_obs = hi_act_obs.copy()
                    last_hi_a = hi_action.copy()
                    last_hi_r = 0.
                    done = False
                    if self.old_sample or valid_count:
                        hi_action_for_low = hi_action
                    else:
                        # make hi_action a delta phi(s)
                        hi_action_for_low = ag.copy() + hi_action.copy()
                        hi_action_for_low = np.clip(hi_action_for_low, -SUBGOAL_RANGE, SUBGOAL_RANGE)
                    hi_action_tensor = torch.tensor(hi_action_for_low, dtype=torch.float32).unsqueeze(0).to(self.device)
                    # update high-level policy
                    if len(self.hi_buffer) > self.args.batch_size and self.learn_hi:
                        self.update_hi(epoch)
                with torch.no_grad():
                    if self.not_train_low:
                        action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                    else:
                        action = self.explore_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                # feed the actions into the environment
                observation_new, r, _, info = self.env.step(action)
                if info['is_success']:
                    done = True
                    # only record the first success
                    if success == 0 and is_furthest_task:
                        success = t
                        self.train_success += 1
                if self.animate:
                    self.env.render()
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal'][:self.real_goal_dim]
                ag_record_new = observation_new['achieved_goal']
                if self.learn_goal_space:
                    if self.args.gradient_flow:
                        if self.args.use_target:
                            ag_new = self.hi_agent.policy_target.phi(
                                torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()
                        else:
                            ag_new = self.hi_agent.policy.phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()
                    elif self.args.gradient_flow_value:
                        ag_new = self.hi_agent.critic.phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                    elif self.args.use_prediction:
                        ag_new = self.representation.phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                    else:
                        if self.args.use_target:
                            ag_new = self.target_phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                        else:
                            ag_new = self.representation(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                if self.whole_obs:
                    ag_new = obs_new.copy()
                if done is False:
                    # counting after starting to update phi (updating phi every 100 episodes)
                    if self.count_latent and epoch > self.phi_interval:
                        intrinsic_rewards = self.hash.predict_rewards(ag[None])[0] * self.intrinsic_coeff
                        r += intrinsic_rewards
                        self.hash.inc_hash(ag.copy()[None])
                        self.xy_hash.inc_hash(ag_record.copy()[None])
                    if self.count_obs:
                        self.hash.inc_hash(obs[None])
                        r += self.hash.predict(obs_new[None])[0] * 0.1
                    last_hi_r += r
                # append rollouts
                ep_obs.append(obs[:self.low_dim].copy())
                ep_ag.append(ag.copy())
                ep_ag_record.append(ag_record.copy())
                ep_g.append(hi_action_for_low.copy())
                ep_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new
                ag_record = ag_record_new
                # # slowly update phi
                # if epoch > self.start_update_phi and not self.not_update_phi and not self.args.gradient_flow and not self.args.gradient_flow_value:
                #     self.slow_update_phi(epoch)
                #     if t % self.args.period == 0 and self.args.use_target:
                #         self._soft_update_target_network(self.target_phi, self.representation)
            ep_obs.append(obs[:self.low_dim].copy())
            ep_ag.append(ag.copy())
            ep_ag_record.append(ag_record.copy())
            mask = float(not done)
            hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
            self.hi_buffer.push(last_hi_obs, last_hi_a, last_hi_r, hi_act_obs, mask, epoch)

            mb_obs = np.array([ep_obs])
            mb_ag = np.array([ep_ag])
            mb_ag_record = np.array([ep_ag_record])
            mb_g = np.array([ep_g])
            mb_actions = np.array([ep_actions])
            self.low_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, success, False, mb_ag_record])
            if self.count_latent and epoch > self.phi_interval:
                if not self.usual_update_hash:
                    self.hash.inc_hash(ag.copy()[None])
                self.xy_hash.inc_hash(ag_record.copy()[None])

            if self.args.save and self.args.env_name == "NChain-v1":
                self.writer.add_scalar('Explore/coverage_' + self.args.env_name, self.env.env.coverage, epoch)

            # update low-level
            if not self.not_train_low:
                for n_batch in range(self.args.n_batches):
                    self._update_network(epoch, self.low_buffer, self.low_actor_target_network,
                                         self.low_critic_target_network,
                                         self.low_actor_network, self.low_critic_network, 'max_timesteps',
                                         self.low_actor_optim, self.low_critic_optim, use_forward_loss=self.low_forward, clip=self.low_use_clip)
                    if n_batch % self.args.period == 0:
                        self._soft_update_target_network(self.low_actor_target_network, self.low_actor_network)
                        self._soft_update_target_network(self.low_critic_target_network, self.low_critic_network)

            # very very slow to learn phi
            if epoch > self.start_update_phi and not self.not_update_phi and epoch % self.phi_interval == 0:
                for _ in range(50000):
                    self.slow_update_phi(epoch)


            # start to do the evaluation
            if epoch % self.args.eval_interval == 0 and epoch != 0:
                # calculate train success rate
                train_success_rate = self.train_success / self.args.eval_interval * 10
                self.count_prob = np.exp(1 - train_success_rate) / (np.exp(1 - train_success_rate) + np.exp(train_success_rate))
                self.train_SR = train_success_rate / 10
                self.train_success = 0
                if not self.learn_hi:
                    self.eval_trajectory(epoch)
                if self.test_env1 is not None:
                    eval_success1, _ = self._eval_hier_agent(env=self.test_env1)
                    eval_success2, _ = self._eval_hier_agent(env=self.test_env2)
                farthest_success_rate, _ = self._eval_hier_agent(env=self.test_env)
                random_success_rate, _ = self._eval_hier_agent(env=self.env)
                self.success_log.append(farthest_success_rate)
                mean_success = np.mean(self.success_log[-5:])
                # stop updating phi and low
                if self.early_stop and (mean_success >= 0.9 or epoch > self.early_stop_thres):
                    print("early stop !!!")
                    self.not_update_phi = True
                    self.not_train_low = True
                print('[{}] epoch is: {}, eval hier success rate is: {:.3f}'.format(datetime.now(), epoch, random_success_rate))
                if self.save_fig:
                    self.vis_hier_policy(epoch=epoch)
                    self.visualize_representation(epoch=epoch)
                if self.args.save:
                    print("log_dir: ", self.log_dir)
                    self.plot_exploration(epoch)
                    torch.save([self.hi_agent.critic.state_dict()], self.model_path + '/hi_critic_model.pt')
                    torch.save([self.low_critic_network.state_dict()], self.model_path + '/low_critic_model.pt')
                    torch.save(self.hi_buffer, self.model_path + '/hi_buffer.pt')
                    if not self.args.gradient_flow and not self.args.gradient_flow_value:
                        if self.save_model:
                            # self.cal_MIV(epoch)
                            torch.save([self.representation.state_dict()], self.model_path + '/phi_model_{}.pt'.format(epoch))
                            torch.save([self.hi_agent.policy.state_dict()], self.model_path + '/hi_actor_{}.pt'.format(epoch))
                            torch.save([self.low_actor_network.state_dict()], self.model_path + '/low_actor_{}.pt'.format(epoch))
                            torch.save(self.low_buffer, self.model_path + '/low_buffer.pt')
                        else:
                            torch.save([self.representation.state_dict()], self.model_path + '/phi_model.pt')
                            torch.save([self.hi_agent.policy.state_dict()], self.model_path + '/hi_actor_model.pt')
                            torch.save([self.low_actor_network.state_dict()], self.model_path + '/low_actor_model.pt')
                            torch.save(self.low_buffer, self.model_path + '/low_buffer.pt')
                    self.writer.add_scalar('Success_rate/hier_farthest_' + self.args.env_name, farthest_success_rate, epoch)
                    self.writer.add_scalar('Success_rate/hier_random_' + self.args.env_name, random_success_rate, epoch)
                    self.writer.add_scalar('Success_rate/train_' + self.args.env_name, self.train_SR, epoch)
                    self.writer.add_scalar('Explore/furthest_task_' + self.args.env_name, self.furthest_task, epoch)
                    self.writer.add_scalar('Explore/Valid_' + self.args.env_name, self.valid_times, epoch)
                    if self.test_env1 is not None:
                        self.writer.add_scalar('Success_rate/eval1_' + self.args.env_name,
                                               eval_success1, epoch)
                        self.writer.add_scalar('Success_rate/eval2_' + self.args.env_name, eval_success2,
                                               epoch)
                self.count_xy_record = []
                self.subgoal_record = []

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
        return obs, g

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        if action.shape == ():
            action = np.array([action])
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        if np.random.rand() < self.args.random_eps:
            action = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                       size=self.env_params['action'])
        return action

    def explore_policy(self, obs, goal):
        pi = self.low_actor_network(obs, goal)
        action = self._select_actions(pi)
        return action

    def update_hi(self, epoch):
        if self.args.gradient_flow or self.args.gradient_flow_value:
            sample_data, _ = self.slow_collect()
            sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
        else:
            sample_data = None
        critic_1_loss, critic_2_loss, policy_loss, _, _ = self.hi_agent.update_parameters(self.hi_buffer,
                                                                                          self.args.batch_size,
                                                                                          self.env_params,
                                                                                          self.hi_sparse,
                                                                                          sample_data)
        if self.args.save:
            self.writer.add_scalar('Loss/hi_critic_1', critic_1_loss, epoch)
            self.writer.add_scalar('Loss/hi_critic_2', critic_2_loss, epoch)
            self.writer.add_scalar('Loss/hi_policy', policy_loss, epoch)

    def random_policy(self, obs, goal):
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        return random_actions

    def test_policy(self, obs, goal):
        pi = self.low_actor_network(obs, goal)
        # convert the actions
        actions = pi.detach().cpu().numpy().squeeze()
        if actions.shape == ():
            actions = np.array([actions])
        return actions

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self, epoch, buffer, actor_target, critic_target, actor, critic, T, actor_optim, critic_optim, use_forward_loss=True, clip=True):
        # sample the episodes
        transitions = buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g, ag = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['ag']
        transitions['obs'], transitions['g'] = o, g
        transitions['obs_next'], transitions['g_next'] = o_next, g
        ag_next = transitions['ag_next']

        # start to do the update
        obs_cur = transitions['obs']
        g_cur = transitions['g']
        obs_next = transitions['obs_next']
        g_next = transitions['g_next']

        # done
        dist = np.linalg.norm(ag_next - g_next, axis=1)
        not_done = (dist > self.distance_threshold).astype(np.int32).reshape(-1, 1)

        # transfer them into the tensor
        obs_cur = torch.tensor(obs_cur, dtype=torch.float32).to(self.device)
        g_cur = torch.tensor(g_cur, dtype=torch.float32).to(self.device)
        obs_next = torch.tensor(obs_next, dtype=torch.float32).to(self.device)
        g_next = torch.tensor(g_next, dtype=torch.float32).to(self.device)
        ag_next = torch.tensor(ag_next, dtype=torch.float32).to(self.device)
        not_done = torch.tensor(not_done, dtype=torch.int32).to(self.device)

        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32).to(self.device)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32).to(self.device)

        # calculate the target Q value function
        with torch.no_grad():
            actions_next = actor_target(obs_next, g_next)
            q_next_value = critic_target(obs_next, g_next, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + critic_target.gamma * q_next_value * not_done
            target_q_value = target_q_value.detach()
            if clip:
                clip_return = self.env_params[T]
                target_q_value = torch.clamp(target_q_value, -clip_return, 0.)
        # the q loss
        real_q_value = critic(obs_cur, g_cur, actions_tensor)

        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # add a L2 norm loss to the critic loss
        L2_reg = torch.tensor(0., requires_grad=True).to(self.args.device)
        for name, param in critic.named_parameters():
            L2_reg = L2_reg + torch.norm(param)
        critic_loss += 0. * L2_reg

        if use_forward_loss:
            forward_loss = critic(obs_cur, ag_next, actions_tensor).pow(2).mean()
            critic_loss += forward_loss
        # the actor loss
        actions_real = actor(obs_cur, g_cur)
        actor_loss = -critic(obs_cur, g_cur, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()

        # start to update the network
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()
        # update the critic_network
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        # # debug low-level diverge
        # if critic_loss.detach() > 10.:
        #     print("#" * 20)
        #     print("real_q", real_q_value)
        #     print("r_tensor", r_tensor)
        #     print("q_next", q_next_value)


        if self.args.save:
            if T == 'max_timesteps':
                name = 'low'
            else:
                name = 'high'
            self.writer.add_scalar('Loss/' + name + '_actor_loss' + self.args.metric, actor_loss, epoch)
            self.writer.add_scalar('Loss/' + name + '_critic_loss' + self.args.metric, critic_loss, epoch)
            with torch.no_grad():
                target_q_mean = target_q_value.mean()
                real_q_mean = real_q_value.detach().mean()
                r_mean = r_tensor.detach().mean()
                q_next_mean = q_next_value.mean()
                L2_mean = L2_reg.detach().mean()
                self.writer.add_scalar('Loss/' + name + '_target_q', target_q_mean, epoch)
                self.writer.add_scalar('Loss/' + name + '_real_q', real_q_mean, epoch)
                self.writer.add_scalar('Loss/' + name + '_r', r_mean, epoch)
                self.writer.add_scalar('Loss/' + name + '_target_q_next', q_next_mean, epoch)
                self.writer.add_scalar('Loss/' + name + '_weights_l2', L2_mean, epoch)

    def _eval_hier_agent(self, env, n_test_rollouts=10):
        total_success_rate = []
        if not self.args.eval:
            n_test_rollouts = self.args.n_test_rollouts
        discount_reward = np.zeros(n_test_rollouts)
        for roll in range(n_test_rollouts):
            per_success_rate = []
            observation = env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for num in range(self.env_params['max_test_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    if num % self.c == 0:
                        hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
                        hi_action = self.hi_agent.select_action(hi_act_obs, evaluate=True)
                        if self.old_sample:
                            new_hi_action = hi_action
                        else:
                            ag = self.representation(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                            new_hi_action = ag + hi_action
                            new_hi_action = np.clip(new_hi_action, -SUBGOAL_RANGE, SUBGOAL_RANGE)
                        hi_action_tensor = torch.tensor(new_hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                    action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                observation_new, rew, done, info = env.step(action)
                if self.animate:
                    env.render()
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                if done:
                    per_success_rate.append(info['is_success'])
                    if bool(info['is_success']):
                        # print("t:", num)
                        discount_reward[roll] = 1 - 1. / self.env_params['max_test_timesteps'] * num
                    break
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        global_reward = np.mean(discount_reward)
        if self.args.eval:
            print("hier success rate", global_success_rate, global_reward)
        return global_success_rate, global_reward

    def init_network(self):
        self.low_actor_network = actor(self.env_params, self.real_goal_dim).to(self.device)
        self.low_actor_target_network = actor(self.env_params, self.real_goal_dim).to(self.device)
        self.low_critic_network = criticWrapper(self.env_params, self.args, self.real_goal_dim).to(self.device)
        self.low_critic_target_network = criticWrapper(self.env_params, self.args, self.real_goal_dim).to(self.device)

        self.start_epoch = 0

        # create the optimizer
        self.low_actor_optim = torch.optim.Adam(self.low_actor_network.parameters(), lr=self.args.lr_actor)
        self.low_critic_optim = torch.optim.Adam(self.low_critic_network.parameters(), lr=self.args.lr_critic, weight_decay=self.args.weight_decay)

    def sync_target(self):
        # load the weights into the target networks
        self.low_actor_target_network.load_state_dict(self.low_actor_network.state_dict())
        self.low_critic_target_network.load_state_dict(self.low_critic_network.state_dict())

    def slow_update_phi(self, epoch):
        sample_data, hi_action = self.slow_collect()
        sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
        if not self.args.use_prediction:
            obs, obs_next = self.representation(sample_data[0]), self.representation(sample_data[1])
            min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
            hi_obs, hi_obs_next = self.representation(sample_data[2]), self.representation(sample_data[3])
            max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
            representation_loss = (min_dist + max_dist).mean()
            # add l2 regularization
            representation_loss += self.feature_reg * (obs / self.abs_range).pow(2).mean()
        else:
            hi_action = torch.tensor(hi_action, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                target_next_obs = self.representation.phi(sample_data[3])
            obs, obs_next = self.representation.phi(sample_data[0]), self.representation.phi(sample_data[1])
            min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
            hi_obs, hi_obs_next = self.representation.phi(sample_data[2]), self.representation.phi(sample_data[3])
            max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
            representation_loss = (min_dist + max_dist).mean()
            # prediction loss
            if self.use_prob:
                predict_distribution = self.representation(sample_data[2], hi_action)
                prediction_loss = - predict_distribution.log_prob(target_next_obs).mean()
            else:
                predict_state = self.representation(sample_data[2], hi_action)
                prediction_loss = (predict_state - target_next_obs).pow(2).mean()
            representation_loss += self.prediction_coeff * prediction_loss
        self.representation_optim.zero_grad()
        representation_loss.backward()
        self.representation_optim.step()
        if self.args.save:
            self.writer.add_scalar('Loss/phi_loss' + self.args.metric, representation_loss, epoch)

    def slow_collect(self, batch_size=100):
        if self.args.use_prediction:
            transitions = self.low_buffer.sample(batch_size)
            obs, obs_next = transitions['obs'], transitions['obs_next']

            hi_obs, hi_action, _, hi_obs_next, _ = self.hi_buffer.sample(batch_size)
            hi_obs, hi_obs_next = hi_obs[:, :self.env_params['obs']], hi_obs_next[:, :self.env_params['obs']]
            train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
            return train_data, hi_action
        else:
            # new negative samples
            episode_num = self.low_buffer.current_size
            obs_array = self.low_buffer.buffers['obs'][:episode_num]
            episode_idxs = np.random.randint(0, episode_num, batch_size)
            t_samples = np.random.randint(self.env_params['max_timesteps'] - self.k - self.delta_k, size=batch_size)
            if self.delta_k > 0:
                delta = np.random.randint(self.delta_k, size=batch_size)
            else:
                delta = 0

            hi_obs = obs_array[episode_idxs, t_samples]
            hi_obs_next = obs_array[episode_idxs, t_samples + self.k + delta]
            obs = hi_obs
            obs_next = obs_array[episode_idxs, t_samples + 1 + delta]

            # filter data when the robot is ant
            if self.args.env_name[:3] == "Ant":
                good_index = np.where((hi_obs[:, 2] >= 0.3) & (hi_obs_next[:, 2] >= 0.3) & (obs_next[:, 2] >= 0.3))[0]
                hi_obs = hi_obs[good_index]
                hi_obs_next = hi_obs_next[good_index]
                obs = hi_obs
                obs_next = obs_next[good_index]
                assert len(hi_obs) == len(hi_obs_next) == len(obs_next)

            train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
            return train_data, None

    def visualize_representation(self, epoch):
        transitions = self.low_buffer.sample(800)
        obs = transitions['obs']
        # with open('fig/final/' + "sampled_states.pkl", 'wb') as output:
        #     pickle.dump(obs, output)

        index1 = np.where((obs[:, 0] < 4) & (obs[:, 1] < 4))
        index2 = np.where((obs[:, 0] < 4) & (obs[:, 1] > 4))
        index3 = np.where((obs[:, 0] > 4) & (obs[:, 1] < 4))
        index4 = np.where((obs[:, 0] > 4) & (obs[:, 1] > 4))
        index_lst = [index1, index2, index3, index4]

        obs_tensor = torch.Tensor(obs).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        plt.scatter(features[:, 0], features[:, 1], color='green')
        plt.show()

        # rep = []
        # for index in index_lst:
        #     rep.append(features[index])
        #
        # self.plot_fig(rep, 'slow_feature', epoch)
        #
        #
        # obs_list = []
        # for index in index_lst:
        #     obs_list.append(obs[index])
        # self.plot_fig(obs_list, 'obs', epoch)

        '''
        tsne_list = []
        res_tsne = TSNE(n_components=2).fit_transform(obs)
        for index in index_lst:
            tsne_list.append(res_tsne[index])
        self.plot_fig(tsne_list, 'tsne_feature', epoch)
        '''

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
        plt.savefig('fig/final/' + name + str(epoch) + '.png')
        plt.close()

    def vis_hier_policy(self, epoch=0, load_obs=None, path=None):
        obs_vec = []
        hi_action_vec = []
        env = self.test_env
        observation = env.reset()
        obs = observation['observation']
        ag_record = observation['achieved_goal']
        obs_vec.append(obs.copy())
        if self.args.image:
            obs_vec[-1][:2] = ag_record
        g = observation['desired_goal']
        if load_obs is None:
            for num in range(self.env_params['max_test_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    if num % self.c == 0:
                        hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
                        hi_action = self.hi_agent.select_action(hi_act_obs, evaluate=True)
                        hi_action_tensor = torch.tensor(hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                        hi_action_vec.append(hi_action)
                    action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                observation_new, rew, done, info = env.step(action)
                if self.animate:
                    env.render()
                obs = observation_new['observation']
                ag_record = observation_new['achieved_goal']
                obs_vec.append(obs.copy())
                if self.args.image:
                    obs_vec[-1][:2] = ag_record
                if done:
                    if info['is_success']:
                        print("success !!!")
                    break
        else:
            obs_vec = load_obs

        plt.figure(figsize=(12, 6))
        obs_vec = np.array(obs_vec)
        with open('fig/final/' + "maze_rollout.pkl", 'wb') as output:
            pickle.dump(obs_vec, output)
        self.plot_rollout(obs_vec, "XY_{}".format(epoch * self.env_params['max_timesteps']), 121, goal=g)

        if not self.learn_goal_space:
            features = obs_vec[:, :2]
            feature_goal = g[:2]
        else:
            if self.args.image:
                obs_vec[:, :2] = 0.
            obs_tensor = torch.Tensor(obs_vec[:, :self.hi_dim]).to(self.device)
            features = self.representation(obs_tensor).detach().cpu().numpy()
            rest = (self.env_params['obs'] - self.env_params['goal']) * [0.]
            g = np.concatenate((g, np.array(rest)))
            g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
            feature_goal = self.representation(g).detach().cpu().numpy()[0]
        hi_action_vec = np.array(hi_action_vec)
        if load_obs is None:
            self.plot_rollout(features, "Feature_{}".format(epoch * self.env_params['max_timesteps']), 122, feature_goal, hi_action_vec)
        else:
            self.plot_rollout(features, "Feature_{}".format(epoch * self.env_params['max_timesteps']), 122,
                              feature_goal)
        if path is None:
            file_name = 'fig/round2/rollout' + str(epoch) + '.png'
        else:
            file_name = path + '/rollout_' + str(epoch) + '.png'
        plt.savefig(file_name, bbox_inches='tight', transparent=False)
        # plt.show()
        plt.close()

    def eval_trajectory(self, epoch):
        plt.figure(figsize=(12, 6))
        obs_vec = np.array(self.trajectory)
        self.plot_rollout(obs_vec, "XY_{}".format(epoch * self.env_params['max_timesteps']), 121)


        obs_tensor = torch.Tensor(obs_vec[:, :self.hi_dim]).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        self.plot_rollout(features, "Feature_{}".format(epoch * self.env_params['max_timesteps']), 122)

        file_name = 'fig/latent1/rollout' + str(epoch) + '.png'
        plt.savefig(file_name, bbox_inches='tight')
        # plt.show()
        plt.close()

    def plot_rollout(self, obs_vec, name, num, goal=None, hi_action_vec=None, no_axis=False):
        plt.subplot(num)
        cm = plt.cm.get_cmap('RdYlBu')
        num = np.arange(obs_vec.shape[0])
        plt.scatter(obs_vec[:, 0], obs_vec[:, 1], c=num, cmap=cm)
        if goal is not None:
            plt.scatter([goal[0]], [goal[1]], marker='*',
                        color='green', s=200, label='goal')
        if hi_action_vec is not None:
            plt.scatter(hi_action_vec[:, 0], hi_action_vec[:, 1], c="k")
        plt.title(name, fontsize=24)
        if no_axis:
            plt.axis('off')
        if not no_axis:
            plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
                        color='green', s=200, label='start')
            plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
                        color='red', s=200, label='end')
            plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), fontsize=14, borderaxespad=0.)
        # plt.show()

    def cal_MIV(self, epoch):
        log = []
        start_dim = 0
        oracle_dim = 2
        print("start, end", start_dim, oracle_dim)
        for _ in range(5):
            transitions = self.low_buffer.sample(100)
            obs = transitions['obs']

            # hi_obs, _, _, _, _ = self.hi_buffer.sample(100)
            # obs = hi_obs[:, :self.env_params['obs']]

            ora_pos, ora_neg, rest_pos, rest_neg = obs.copy(), obs.copy(), obs.copy(), obs.copy()

            ora_pos[:, start_dim:oracle_dim] = ora_pos[:, start_dim:oracle_dim] * 1.1
            ora_neg[:, start_dim:oracle_dim] = ora_neg[:, start_dim:oracle_dim] * 0.9

            rest_pos[:, oracle_dim:] = rest_pos[:, oracle_dim:] * 1.1
            rest_neg[:, oracle_dim:] = rest_neg[:, oracle_dim:] * 0.9

            oracle = self.cal_IV(ora_pos, ora_neg)
            rest = self.cal_IV(rest_pos, rest_neg)

            MIV_ratio = oracle / (oracle + rest)
            log.append(MIV_ratio)
            if self.args.save:
                self.writer.add_scalar('Success_rate/MIV_ratio', MIV_ratio, epoch)
        arr_mean = np.mean(log)
        arr_std = np.std(log, ddof=1)
        print("mean, std", arr_mean, arr_std)

    def cal_IV(self, pos, neg):
        pos = torch.tensor(pos, dtype=torch.float32).to(self.device)
        neg = torch.tensor(neg, dtype=torch.float32).to(self.device)
        if not self.args.gradient_flow:
            pos_feature = self.representation(pos).detach().cpu().numpy()
            neg_feature = self.representation(neg).detach().cpu().numpy()
        else:
            pos_feature = self.hi_agent.policy.phi(pos).detach().cpu().numpy()
            neg_feature = self.hi_agent.policy.phi(neg).detach().cpu().numpy()
        difference = np.linalg.norm(pos_feature - neg_feature, axis=1)
        return np.mean(difference)

    def vis_learning_process(self, filename, epoch=0):
        # plt.figure(figsize=(15, 6))
        if epoch == 0:
            with open('fig/trajectory/' + filename, 'rb') as output:
                obs_vec = pickle.load(output)
            for i in range(50, 5000, 50):
                self.representation.load_state_dict(torch.load(self.args.resume_path + \
                                                               '/phi_model_{}.pt'.format(i), map_location='cuda:1')[0])
                self.vis_hier_policy(epoch=i, load_obs=obs_vec)
        else:
            self.vis_hier_policy(epoch=epoch, load_obs=self.trajectory, path=self.fig_path)


    def multi_eval(self):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = 'runs/hier/' + str(self.args.env_name) + '/' + current_time + \
                       "_C_" + str(self.args.c) + "_Image_" + str(self.args.image) + \
                       "_Seed_" + str(self.args.seed) + "_Abs_" + str(self.args.abs_range) + \
                       "_Subgoal_" + str(SUBGOAL_RANGE) + "_Early_" + str(self.early_stop_thres) + "_Old_" + str(
            self.old_sample) + "_Res_5" + "_NoPhi_" + str(self.not_update_phi) + "_LearnG_" + str(self.learn_goal_space)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        for i in range(50, 8050, 50):
            try:
                self.hi_agent.policy.load_state_dict(torch.load(self.args.resume_path + \
                                                                '/hi_actor_{}.pt'.format(i), map_location='cuda:0')[0])
                self.low_actor_network.load_state_dict(torch.load(self.args.resume_path + \
                                                                  '/low_actor_{}.pt'.format(i), map_location='cuda:0')[0])
            except:
                print("Epoch:", i, " No such file !!!")
            if self.test_env1 is not None:
                eval_success1, _ = self._eval_hier_agent(env=self.test_env1)
                self.writer.add_scalar('Success_rate/eval1_' + self.args.env_name,
                                       eval_success1, i)
            eval_success, _ = self._eval_hier_agent(env=self.test_env)
            self.writer.add_scalar('Success_rate/hier_farthest_' + self.args.env_name, eval_success, i)

    def same_data_compare(self):
        with open('fig/final/' + "sampled_raw_states.pkl", 'rb') as output:
            Obs = pickle.load(output)

        features = []
        for obs in Obs:
            obs_tensor = torch.Tensor(obs[:, :29]).to(self.device)
            feature = self.representation(obs_tensor).detach().cpu().numpy()
            features.append(feature)

        self.plot_fig(Obs, 'obs', "raw")
        self.plot_fig(features, 'slow_feature', 'slow')

        with open('fig/final/' + "oracle_trajectory.pkl", 'rb') as output:
            obs_vec = pickle.load(output)

        xy = obs_vec[:, :2]
        dists = np.linalg.norm(xy - [[0, 8]], axis=1)
        success = np.argwhere(dists < 1.5)
        min_success = min(success)[0]
        print("min_success", min_success)
        print("min_dist", dists[min_success])

        plt.figure(figsize=(12, 6))
        self.plot_rollout(obs_vec[:min_success], "XY", 121, no_axis=True)

        obs_tensor = torch.Tensor(obs_vec[:min_success, :29]).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        rest = (self.env_params['obs'] - self.real_goal_dim) * [0.]
        g = np.concatenate(([0, 8], np.array(rest)))
        g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
        feature_goal = self.representation(g).detach().cpu().numpy()[0]
        self.plot_rollout(features, "slow feature", 122, no_axis=True)

        file_name = 'fig/final/rollout_compare.png'
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

    def picvideo(self, path, size):

        filenames = []
        for i in range(50, 8000, 50):
            filename = "fig/final/rollout{}.png".format(i)
            filenames.append(filename)

        frames = []
        for image_name in filenames:
            frames.append(imageio.imread(image_name))

        # druation : ?????????????????????????????????
        imageio.mimsave('fig/final/push1.gif', frames, 'GIF', duration=0.3)

    def cal_stable(self):
        transitions = self.low_buffer.sample(100)
        obs = transitions['obs']

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        self.representation.load_state_dict(torch.load(self.args.resume_path + \
                                                       '/phi_model_3000.pt', map_location='cuda:1')[0])

        feature1 = self.representation(obs).detach().cpu().numpy()

        self.representation.load_state_dict(torch.load(self.args.resume_path + \
                                                       '/phi_model_4000.pt', map_location='cuda:1')[0])

        feature2 = self.representation(obs).detach().cpu().numpy()
        distance = np.linalg.norm(feature1 - feature2)
        print("distance", distance)

    def plot_chain(self):
        state = np.eye(40)
        obs_tensor = torch.Tensor(state).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        plt.plot(features)
        plt.show()

    def cal_slow(self):
        num = 5
        record = np.zeros(num)
        state_lst = []
        for i in range(num):
            sample_data, hi_action = self.slow_collect()
            state_lst.append(sample_data[:2])
            print("sample", sample_data[:2].shape)
            sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
            obs, obs_next = self.representation(sample_data[0]), self.representation(sample_data[1])
            min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
            mean_dist = np.mean(min_dist.detach().cpu().numpy())
            record[i] = mean_dist

            print("mean_dist", mean_dist)

        arr_mean = np.mean(record)
        arr_std = np.std(record, ddof=1)
        print(self.args.env_name, arr_mean, arr_std)

        with open('fig/final/' + "img_push_states.pkl", 'wb') as output:
            pickle.dump(state_lst, output)

    def cal_random_slow(self):
        num = 5
        record = np.zeros(num)
        print("hi", self.hi_dim)
        # select_dim = np.random.randint(self.hi_dim, size=2)
        for i in range(num):
            select_dim = np.random.randint(self.hi_dim, size=2)
            sample_data, hi_action = self.slow_collect()
            # print("sample0", sample_data[0])
            obs = sample_data[0][:, select_dim]
            obs_next = sample_data[1][:, select_dim]
            print("new", obs.shape)

            min_dist = np.square(obs - obs_next)
            mean_dist = np.mean(min_dist)
            record[i] = mean_dist

            print("mean_dist", mean_dist)

        arr_mean = np.mean(record)
        arr_std = np.std(record, ddof=1)
        print(self.args.env_name, arr_mean, arr_std)

    def plot_exploration(self, epoch=0):
        matplotlib.rcParams['figure.figsize'] = [10, 10]  # for square canvas
        matplotlib.rcParams['figure.subplot.left'] = 0
        matplotlib.rcParams['figure.subplot.bottom'] = 0
        matplotlib.rcParams['figure.subplot.right'] = 1
        matplotlib.rcParams['figure.subplot.top'] = 1
        def construct_maze(maze_id='Maze'):
            if maze_id == 'Push':
                structure = [
                    [1, 1, 1, 1, 1],
                    [1, 0, 'r', 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 1],
                ]
            elif maze_id == 'Maze1':
                structure = [
                    [1, 1, 1, 1, 1],
                    [1, 'r', 0, 0, 1],
                    [1, 1, 1, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1],
                ]
            else:
                raise NotImplementedError('The provided MazeId %s is not recognized' % maze_id)

            return structure

        def plot_map():
            walls = construct_maze(maze_id="Maze1")
            contain_r = [1 if "r" in row else 0 for row in walls]
            row_r = contain_r.index(1)
            col_r = walls[row_r].index("r")
            walls[row_r][col_r] = 0
            walls = np.array(walls)
            # print("walls", walls)
            scaling = self.scaling
            walls = resize_walls(walls, scaling)
            plot_walls(walls, scaling, row_r, col_r)

        def plot_walls(walls, scaling, row_r, col_r):
            for (i, j) in zip(*np.where(walls)):
                x = np.array([j, j + 1]) - (col_r + 0.5) * scaling
                y0 = np.array([i, i]) - (row_r + 0.5) * scaling
                y1 = np.array([i + 1, i + 1]) - (row_r + 0.5) * scaling
                plt.fill_between(x, y0, y1, color='grey')

        def resize_walls(walls, factor):
            """Increase the environment by rescaling.

            Args:
              walls: 0/1 array indicating obstacle locations.
              factor: (int) factor by which to rescale the environment."""
            (height, width) = walls.shape
            row_indices = np.array([i for i in range(height) for _ in range(factor)])
            col_indices = np.array([i for i in range(width) for _ in range(factor)])
            walls = walls[row_indices]
            walls = walls[:, col_indices]
            assert walls.shape == (factor * height, factor * width)
            return walls

        def save_figs(logdir, i, crop):
            fig = plt.figure(figsize=(18, 6))
            transitions = self.low_buffer.sample(1000)
            if 'ag_record' in transitions.keys():
                ag_real = transitions['ag_record'][:, :2]
            else:
                ag_real = transitions['obs'][:, :2]

            # fig1
            ax = fig.add_subplot(1, 3, 1)
            plot_map()
            cm = plt.cm.get_cmap('RdYlBu')
            if self.count_latent:
                count_xy = np.array(self.xy_hash.predict(ag_real)).astype(int)
                plt.scatter(ag_real[:, 0], ag_real[:, 1], c=count_xy, cmap=cm, s=30, alpha=0.5, label='explored area')
                # plt.colorbar()
            else:
                plt.scatter(ag_real[:, 0], ag_real[:, 1], color='green', s=30, alpha=0.2, label='explored area')
            plt.title('XY', fontsize=24)

            # fig2
            ax = fig.add_subplot(1, 3, 2)
            obs = transitions['obs']
            obs_tensor = torch.Tensor(obs).to(self.device)
            features = self.representation(obs_tensor).detach().cpu().numpy()
            if self.count_latent:
                count = np.array(self.hash.predict(features)).astype(int)
                plt.scatter(features[:, 0], features[:, 1], c=count, cmap=cm, s=30, alpha=0.5, label='explored area')
                # plt.colorbar()
                intrinsic_rewards = np.array(self.hash.predict_rewards(features)) * self.intrinsic_coeff
                for num in range(int(0.05 * len(intrinsic_rewards))):
                    print_str = "%.2f" % intrinsic_rewards[num]
                    ax.text(features[num, 0], features[num, 1], print_str, fontsize=10, color='blue', alpha=0.6)
            else:
                plt.scatter(features[:, 0], features[:, 1], color='green', s=30, alpha=0.2, label='explored area')

            plt.legend(loc=2, fontsize=14, borderaxespad=0.)
            plt.title('Intrinsic Rewards in Feature Space', fontsize=24)

            # figure 3
            ax = fig.add_subplot(1, 3, 3)
            if self.count_latent:
                delta_count = count_xy - count
                xy_larger = np.where(delta_count > 0)[0]
                plt.scatter(features[xy_larger, 0], features[xy_larger, 1], color='k', s=30, alpha=0.2,
                            label='xy > feature')
                xy_smaller = np.where(delta_count <= 0)[0]
                plt.scatter(features[xy_smaller, 0], features[xy_smaller, 1], color='tab:orange', s=30, alpha=0.2,
                            label='xy <= feature')
                for num in range(int(0.05 * len(features))):
                    print_str = "({}, {})".format(count[num], delta_count[num])
                    if delta_count[num] > 0:
                        color = 'tab:brown'
                    else:
                        color = '#FF00FF'
                    ax.text(features[num, 0], features[num, 1], print_str, fontsize=10, color=color, alpha=0.6)
            plt.legend(loc=0, fontsize=14, borderaxespad=0.)
            plt.title('Feature Count ($\delta$=xy-feature)', fontsize=24)

            plt.savefig(logdir + 'maze_uncropped_' + str(i) + '.png', bbox_inches='tight', transparent=False)
            plt.close()

            if crop:
                img = cv2.imread(logdir + 'push_img_' + str(i) + '.png')
                print(img.shape)
                cropped = img[int(0.2 * img.shape[1]):int(0.78 * img.shape[1]),
                          int(0.23 * img.shape[0]):int(0.8 * img.shape[0])]  # ?????[y0:y1, x0:x1]
                cv2.imwrite(logdir + 'maze_' + str(i) + '.png', cropped)

        if self.args.eval:
            for i in range(0, 100, 50):
                # self.low_buffer = torch.load(self.args.resume_path + '/low_buffer_{}.pt'.format(i), map_location='cuda:1')
                self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')

                save_figs('fig/exp/', i, False)
        else:
            save_figs(self.fig_path + '/', epoch, False)
            self.vis_learning_process(filename=None, epoch=epoch)


    def cal_fall_over(self):
        self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')
        print("load buffer !!!")
        transitions = self.low_buffer.sample(100000)
        obs, ag_record = transitions['obs'], transitions['ag_record']
        select_index = np.where((obs[:, 0] < -1.0) & (obs[:, 1] < -1.0))[0]
        print("select", len(select_index))
        total_count = len(select_index)
        obs_new = obs[select_index]
        not_fall = np.where((obs_new[:, 2] >= 0.3) & (obs_new[:, 2] <= 1.0))[0]
        smaller = np.where(obs_new[:, 2] < 0.3)[0]
        larger = np.where(obs_new[:, 2] > 1.0)[0]
        print("not fall over", len(not_fall))
        not_fall_rate = len(not_fall) / total_count
        print("not fall rate", not_fall_rate)
        print("smaller", len(smaller) / total_count)
        print("larger", len(larger) / total_count)
        print("#" * 20)

        # new negative samples
        batch_size = 10000
        episode_num = self.low_buffer.current_size
        obs_array = self.low_buffer.buffers['obs'][:episode_num]
        episode_idxs = np.random.randint(0, episode_num, batch_size)
        t_samples = np.random.randint(self.env_params['max_timesteps'] - self.k - self.delta_k, size=batch_size)
        if self.delta_k > 0:
            delta = np.random.randint(self.delta_k, size=batch_size)
        else:
            delta = 0

        hi_obs = obs_array[episode_idxs, t_samples]
        hi_obs_next = obs_array[episode_idxs, t_samples + self.k + delta]
        obs = hi_obs
        obs_next = obs_array[episode_idxs, t_samples + 1 + delta]

        # filter data when the robot is ant
        if self.args.env_name[:3] == "Ant":
            good_index = np.where((hi_obs[:, 2] >= 0.3) & (hi_obs_next[:, 2] >= 0.3) & (obs_next[:, 2] >= 0.3))[0]
            hi_obs = hi_obs[good_index]
            hi_obs_next = hi_obs_next[good_index]
            obs = hi_obs
            obs_next = obs_next[good_index]

            select_index = np.where((hi_obs[:, 0] < -1.0) & (hi_obs[:, 1] < -1.0) & (hi_obs_next[:, 0] < -1.0) & (hi_obs_next[:, 1] < -1.0) & \
                                    (obs_next[:, 0] < -1.0) & (obs_next[:, 1] < -1.0))[0]
            hi_obs = hi_obs[select_index]
            hi_obs_next = hi_obs_next[select_index]
            obs = hi_obs
            obs_next = obs_next[select_index]

            c_change = np.linalg.norm(hi_obs - hi_obs_next, axis=1)
            one_change = np.linalg.norm(obs - obs_next, axis=1)
            delta_change = c_change - one_change
            print("c_change", len(c_change))
            print("len_obs", len(hi_obs))
            x_change = np.linalg.norm(hi_obs[:, 0:2] - hi_obs_next[:, 0:2], axis=1) - np.linalg.norm(obs[:, 0:2] - obs_next[:, 0:2], axis=1)
            print("delta_change", delta_change)
            print("x_change", x_change)


        train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])













