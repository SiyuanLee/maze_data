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
import gym
from .PPO.ppo import PPO
from .PPO.replay_buffer import Memory


class hier_ppo_agent:
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

        self.learn_goal_space = False
        self.whole_obs = False  # use whole observation space as subgoal space
        self.direction_goal = False  # if False, use absolute goal
        self.abs_range = abs_range = 25.  # absolute goal range
        self.feature_reg = 0.0  # feature l2 regularization

        if not self.direction_goal:
            if args.env_name[:5] == "Fetch":
                maze_low = self.env.env.initial_gripper_xpos[:2] - self.env.env.target_range
                maze_high = self.env.env.initial_gripper_xpos[:2] + self.env.env.target_range
                self.hi_act_space = gym.spaces.Box(low=maze_low, high=maze_high)
            else:
                self.hi_act_space = self.env.env.maze_space
                d_range = 10.
            if self.learn_goal_space:
                self.hi_act_space = gym.spaces.Box(low=np.array([-abs_range, -abs_range]), high=np.array([abs_range, abs_range]))
            if self.whole_obs:
                vel_low = [-10.] * 4
                vel_high = [10.] * 4
                maze_low = np.concatenate((self.env.env.maze_low, np.array(vel_low)))
                maze_high = np.concatenate((self.env.env.maze_high, np.array(vel_high)))
                self.hi_act_space = gym.spaces.Box(low=maze_low, high=maze_high)
        else:
            d_range = 2.  # direction range, if not adjust, d_range=2


        dense_low = False
        self.low_use_clip = (not dense_low) and (not self.direction_goal)  # only sparse reward use clip
        if args.replay_strategy == "future":
            self.low_forward = True
            assert self.low_use_clip is True
        else:
            self.low_forward = False
            assert self.low_use_clip is False

        # params of learning phi
        resume_phi = args.resume
        self.not_update_phi = True
        self.update_interval = 10  # intervals of updating representation
        phi_path = args.resume_path
        # after max_update, don't update phi
        self.max_update = args.n_epochs
        # self.max_update = 2000

        self.save_fig = False
        self.early_stop = False
        self.success_log = []
        self.least_update = 500  # least episode number to learn phi

        if self.args.save:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.log_dir = 'runs/hier/point/DirectG_' + str(self.direction_goal) + "_" + current_time + \
                            "_Least_" + str(self.least_update) + "_MaxUpdate_" + str(self.max_update) + \
                            "_Save_" + str(self.save_fig) + "_Early_" + str(self.early_stop) + "_LearnG_" + str(self.learn_goal_space)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name + "_" + current_time)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
        # init low-level network
        self.real_goal_dim = 2  # low-level goal space and high-level action space
        self.init_network()
        # init high-level agent
        self.hi_agent = PPO(self.hi_dim, self.real_goal_dim, 0.5, 0.0003, (0.9, 0.999), 0.99, 80, 0.2, self.device, d_range)
        self.hi_buffer = Memory()

        # her sampler
        self.c = self.args.c  # interval of high level action
        self.low_her_module = her_sampler(args.replay_strategy, args.replay_k, args.distance, args.future_step,
                                          dense_reward=dense_low, direction_reward=self.direction_goal)
        self.env_params['real_goal_dim'] = self.real_goal_dim
        if args.env_name[:5] == "Fetch":
            self.low_buffer = replay_buffer_energy(self.env_params, self.args.buffer_size,
                                               self.low_her_module.sample_her_energy, args.env_name)
        else:
            self.low_buffer = replay_buffer(self.env_params, self.args.buffer_size, self.low_her_module.sample_her_transitions)

        not_load_buffer, not_load_high = False, False
        if self.resume is True:
            self.start_epoch = self.resume_epoch
            if not not_load_high:
                self.hi_agent.policy.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/hi_actor_model.pt', map_location='cuda:1')[0])
            self.low_actor_network.load_state_dict(torch.load(self.args.resume_path + \
                                                             '/low_actor_model.pt', map_location='cuda:1')[0])
            self.low_critic_network.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/low_critic_model.pt', map_location='cuda:1')[0])
            if not not_load_buffer:
                self.hi_buffer = torch.load(self.args.resume_path + '/hi_buffer.pt', map_location='cuda:1')
                self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')

        # sync target network of low-level
        self.sync_target()

        if hasattr(self.env.env, 'env'):
            self.animate = self.env.env.env.visualize_goal
        else:
            self.animate = self.args.animate
        self.distance_threshold = self.args.distance

        if args.contrastive_phi:
            self.representation = RepresentationNetwork(env_params, layer=3).to(args.device)
            self.representation_optim = torch.optim.Adam(self.representation.parameters(), lr=0.0001)
            if resume_phi is True:
                self.representation.load_state_dict(torch.load(phi_path + \
                                                               '/phi_model.pt', map_location='cuda:1')[0])

        print("learn goal space", self.learn_goal_space, " not update phi", self.not_update_phi)

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

            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            success = 0
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            if self.learn_goal_space:
                ag = self.representation(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
            if self.whole_obs:
                ag = obs.copy()

            for t in range(self.env_params['max_timesteps']):
                act_obs, act_g = self._preproc_inputs(obs, g)
                with torch.no_grad():
                    if t % self.c == 0:
                        if t != 0:
                            # Saving reward and is_terminals:
                            self.hi_buffer.rewards.append(last_hi_r)
                            self.hi_buffer.is_terminals.append(done)
                        hi_action = self.hi_agent.select_action(obs[:self.hi_dim], self.hi_buffer)
                        last_hi_r = 0.
                        done = False
                        hi_action_tensor = torch.tensor(hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                    if self.not_train_low:
                        action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                    else:
                        action = self.explore_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                # feed the actions into the environment
                observation_new, r, _, info = self.env.step(action)
                if done is False:
                    last_hi_r += r
                if info['is_success']:
                    done = True
                    success = 1
                if self.animate:
                    self.env.render()
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                if self.learn_goal_space:
                    ag_new = self.representation(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                if self.whole_obs:
                    ag_new = obs_new.copy()
                # append rollouts
                ep_obs.append(obs[:self.low_dim].copy())
                ep_ag.append(ag.copy())
                ep_g.append(hi_action.copy())
                ep_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new
            ep_obs.append(obs[:self.low_dim].copy())
            ep_ag.append(ag.copy())
            # Saving reward and is_terminals:
            self.hi_buffer.rewards.append(last_hi_r)
            self.hi_buffer.is_terminals.append(True)

            mb_obs = np.array([ep_obs])
            mb_ag = np.array([ep_ag])
            mb_g = np.array([ep_g])
            mb_actions = np.array([ep_actions])
            self.low_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, success])

            # update high-level
            if epoch % 80 == 0 and epoch != 0:
                self.hi_agent.update(self.hi_buffer)
                self.hi_buffer.clear_memory()

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

            # update phi
            if self.args.contrastive_phi and epoch % self.update_interval == 0 and epoch > 0 and not self.not_update_phi \
                    and epoch < self.max_update:
                self.update_phi(epoch, not_align=False)
                # update achieved goal in low_buffer
                self.low_buffer.update_ag(self.representation, self.device)

            # start to do the evaluation
            if epoch % self.args.eval_interval == 0 and epoch != 0:
                # success_rate = self._eval_agent()
                hier_success_rate, discount_reward = self._eval_hier_agent()
                self.success_log.append(hier_success_rate)
                mean_success = np.mean(self.success_log[-5:])
                # stop updating phi and low
                if self.early_stop and mean_success >= 0.9:
                    self.not_update_phi = True
                    self.not_train_low = True
                print('[{}] epoch is: {}, eval hier success rate is: {:.3f}'.format(datetime.now(), epoch, hier_success_rate))
                if self.save_fig:
                    self.vis_hier_policy(epoch=epoch)
                if self.args.save:
                    print("log_dir: ", self.log_dir)
                    torch.save([self.hi_agent.policy.state_dict()], self.model_path + '/hi_actor_model.pt')
                    torch.save([self.low_critic_network.state_dict()], self.model_path + '/low_critic_model.pt')
                    torch.save([self.low_actor_network.state_dict()], self.model_path + '/low_actor_model.pt')
                    torch.save(self.hi_buffer, self.model_path + '/hi_buffer.pt')
                    torch.save(self.low_buffer, self.model_path + '/low_buffer.pt')
                    # self.writer.add_scalar('Success_rate/low_' + self.args.env_name + self.args.metric, success_rate, epoch)
                    self.writer.add_scalar('Success_rate/hier_' + self.args.env_name + self.args.metric, hier_success_rate, epoch)
                    self.writer.add_scalar('Success_rate/hier_discount_reward', discount_reward, epoch)
                    if self.args.contrastive_phi:
                        torch.save([self.representation.state_dict()], self.model_path + '/phi_model.pt')

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
        return obs, g

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
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

    def random_policy(self, obs, goal):
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        return random_actions

    def test_policy(self, obs, goal):
        pi = self.low_actor_network(obs, goal)
        # convert the actions
        actions = pi.detach().cpu().numpy().squeeze()
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
        if not self.direction_goal:
            dist = np.linalg.norm(ag_next - g_next, axis=1)
            not_done = (dist > self.distance_threshold).astype(np.int32).reshape(-1, 1)
        else:
            a_direction = ag_next - ag  # achieved direction
            cos_dist = np.sum(np.multiply(a_direction, g), axis=1) / ((np.linalg.norm(a_direction, axis=1) * np.linalg.norm(g, axis=1)) + 1e-6)
            not_done = (cos_dist < 0.99).astype(np.int32).reshape(-1, 1)

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
            # clip Q > 0 for direction goal
            if self.direction_goal:
                zero = torch.zeros_like(target_q_value)
                target_q_value = torch.where(target_q_value > 0, zero, target_q_value)
        # the q loss
        real_q_value = critic(obs_cur, g_cur, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
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

        if self.args.save:
            if T == 'max_timesteps':
                name = 'low'
            else:
                name = 'high'
            self.writer.add_scalar('Loss/' + name + '_actor_loss' + self.args.metric, actor_loss, epoch)
            self.writer.add_scalar('Loss/' + name + '_critic_loss' + self.args.metric, critic_loss, epoch)

    # do the evaluation
    def _eval_agent(self, policy=None, n_test_rollouts=100):
        if policy is None:
            policy = self.test_policy

        total_success_rate = []
        if not self.args.eval:
            n_test_rollouts = self.args.n_test_rollouts
        for _ in range(n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    actions = policy(act_obs[:, :self.low_dim], act_g)
                observation_new, _, done, info = self.env.step(actions)
                if self.animate:
                    self.env.render()
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                if done:
                    per_success_rate.append(info['is_success'])
                    break
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        if self.args.eval:
            print("global success rate", global_success_rate)
        return global_success_rate

    # eval low-level policy with absolute goal
    def _eval_low_agent(self, max_length=10, n_test_rollouts=100, vis_rollout=True):
        total_success_rate = 0.
        if vis_rollout:
            n_test_rollouts = 1
        for _ in range(n_test_rollouts):
            observation = self.env.reset()
            obs = observation['observation']
            feature_vec = []
            hi_action = np.array([10, -2])  # subgoal assigned by high-level
            for _ in range(max_length):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, hi_action)
                    actions = self.test_policy(act_obs[:, :self.low_dim], act_g)
                observation_new, _, done, info = self.env.step(actions)
                if self.animate:
                    self.env.render()
                obs = observation_new['observation']
                cur_feature = self.representation(act_obs).detach().cpu().numpy()[0]
                feature_vec.append(cur_feature)
                dist = np.linalg.norm(cur_feature - hi_action)
                if dist < 1:
                    total_success_rate += 1
                    break

        global_success_rate = total_success_rate / n_test_rollouts
        if self.args.eval:
            print("max_length", max_length)
            print("low-level success rate", global_success_rate)
        if vis_rollout:
            feature_vec = np.array(feature_vec)
            self.plot_rollout(feature_vec, "low_rollout", hi_action)

        return global_success_rate

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
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    if num % self.c == 0:
                        if num != 0:
                            # Saving reward and is_terminals:
                            self.hi_buffer.rewards.append(last_hi_r)
                            self.hi_buffer.is_terminals.append(False)
                        hi_action = self.hi_agent.select_action(obs[:self.hi_dim], self.hi_buffer)
                        hi_action_tensor = torch.tensor(hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                        last_hi_r = 0.
                    action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                observation_new, rew, done, info = self.test_env.step(action)
                last_hi_r += rew
                if self.animate:
                    self.test_env.render()
                obs = observation_new['observation']
                if done:
                    per_success_rate.append(info['is_success'])
                    if bool(info['is_success']):
                        discount_reward[roll] = 1 - 1. / self.env_params['max_test_timesteps'] * num
                    break
            # Saving reward and is_terminals:
            self.hi_buffer.rewards.append(last_hi_r)
            self.hi_buffer.is_terminals.append(True)
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
        self.low_critic_optim = torch.optim.Adam(self.low_critic_network.parameters(), lr=self.args.lr_critic)

    def sync_target(self):
        # load the weights into the target networks
        self.low_actor_target_network.load_state_dict(self.low_actor_network.state_dict())
        self.low_critic_target_network.load_state_dict(self.low_critic_network.state_dict())

    def update_phi(self, epoch, not_align=True):
        if not_align:
            # use hi_buffer
            train_data = self.new_collect()
        else:
            # only use low_buffer
            train_data = self.collect(epoch)
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

    def collect(self, epoch):
        all_obs = self.low_buffer.get_all_data()['obs'].copy()
        all_success = self.low_buffer.get_all_data()['success'].copy()
        success_index = np.where(all_success == 1)[0]

        # no enough training data, randomly sample some
        len_success = success_index.shape[0]
        if self.args.save:
            self.writer.add_scalar('Success_rate/train_success', len_success, epoch)
        episode_num = all_success.shape[0]
        least_update = min(episode_num, self.least_update)
        if len_success < least_update:
            select = np.random.choice(episode_num, least_update - len_success, replace=False)
            if len_success > 0:
                success_index = np.concatenate((success_index, select))
            else:
                success_index = select

        obs = all_obs[:, :-1-self.c, :]
        obs_next = all_obs[:, 1:-self.c, :]
        neg_obs = all_obs[:, self.c:-1, :]
        obs, obs_next, neg_obs = obs[success_index], obs_next[success_index], neg_obs[success_index]
        last_shape = obs.shape[2]
        obs, obs_next, neg_obs = obs.reshape(-1, last_shape), obs_next.reshape(-1, last_shape), neg_obs.reshape(-1, last_shape)

        train_data = np.array([obs, obs_next, neg_obs])
        return train_data

    def new_collect(self):
        all_obs = self.low_buffer.get_all_data()['obs'].copy()
        obs = all_obs[:, :-1, :]
        obs_next = all_obs[:, 1:, :]
        last_shape = obs.shape[2]
        obs, obs_next = obs.reshape(-1, last_shape), obs_next.reshape(-1, last_shape)
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

    def vis_rollout(self):
        obs_vec = []
        per_success_rate = []
        observation = self.test_env.reset()
        obs = observation['observation']
        obs_vec.append(obs)
        g = observation['desired_goal']
        for num in range(self.env_params['max_test_timesteps']):
            with torch.no_grad():
                act_obs, act_g = self._preproc_inputs(obs, g)
                if num % self.c == 0:
                    hi_action = self.hi_agent.select_action(obs[:self.hi_dim], evaluate=True)
                    hi_action_tensor = torch.tensor(hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
            observation_new, rew, done, info = self.test_env.step(action)
            if self.animate:
                self.test_env.render()
            obs = observation_new['observation']
            obs_vec.append(obs)
            g = observation_new['desired_goal']
            if done:
                per_success_rate.append(info['is_success'])
                break

        obs_vec = np.array(obs_vec)
        self.plot_rollout(obs_vec, "xy_rollout")

        obs_tensor = torch.Tensor(obs_vec).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        self.plot_rollout(features, "feature_rollout")

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
                act_obs, act_g = self._preproc_inputs(obs, g)
                if num % self.c == 0:
                    hi_action = self.hi_agent.select_action(obs[:self.hi_dim], evaluate=True)
                    hi_action_tensor = torch.tensor(hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                    hi_action_vec.append(hi_action)
                action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
            observation_new, rew, done, info = self.test_env.step(action)
            if self.animate:
                self.test_env.render()
            obs = observation_new['observation']
            obs_vec.append(obs)
            if done:
                per_success_rate.append(info['is_success'])
                break

        obs_vec = np.array(obs_vec)
        self.plot_rollout(obs_vec, "xy_rollout", epoch=epoch)

        obs_tensor = torch.Tensor(obs_vec).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        g = np.concatenate((g, np.array([0., 0., 0., 0.])))
        g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
        feature_goal = self.representation(g).detach().cpu().numpy()[0]
        hi_action_vec = np.array(hi_action_vec)
        self.plot_rollout(features, "hier_rollout", feature_goal, hi_action_vec, epoch)

    def plot_rollout(self, obs_vec, name, goal=None, hi_action_vec=None, epoch=None):
        plt.figure()
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
        if self.args.save:
            fig = plt.gcf()
            self.writer.add_figure('fig/' + name, fig, epoch)
        else:
            if epoch is not None:
                file_name = 'fig/training/' + name + str(epoch) + '.png'
            else:
                file_name = 'fig/' + name + '.png'
            plt.savefig(file_name, bbox_inches='tight')
        plt.close()


