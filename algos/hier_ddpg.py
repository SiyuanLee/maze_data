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



class hier_ddpg_agent:
    def __init__(self, args, env, env_params, test_env):
        self.args = args
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.device = args.device
        self.resume = args.resume
        self.resume_epoch = args.resume_epoch
        self.not_train_high = False
        self.not_train_low = False
        self.shallow = True
        self.low_dim = 6
        self.env_params['low_dim'] = self.low_dim
        if args.hi_dim != 0:
            self.env_params['hi_dim'] = args.hi_dim
        else:
            self.env_params['hi_dim'] = env_params['obs']
        self.hi_dim = self.env_params['hi_dim']
        if args.env_name[:5] == "Fetch":
            maze_low = self.env.env.initial_gripper_xpos[:2] - self.env.env.target_range
            maze_high = self.env.env.initial_gripper_xpos[:2] + self.env.env.target_range
            self.maze_high = np.max(np.concatenate((maze_high, maze_low)))
            self.hi_range = np.array([maze_low, maze_high])
        else:
            self.maze_high = np.max(np.concatenate((self.env.env.maze_high, -self.env.env.maze_low)))
            self.hi_range = np.array([self.env.env.maze_low, self.env.env.maze_high])
        self.hi_random_eps = self.args.hi_random_eps
        self.hi_noise_eps = self.args.noise_eps
        dense_high = False
        dense_low = False
        self.use_clip = not dense_high
        self.low_use_clip = not dense_low
        if args.hi_replay_strategy == "future":
            self.hi_forward = True
            assert dense_high is False
        else:
            self.hi_forward = False
            assert dense_high is True
        if args.replay_strategy == "future":
            self.low_forward = True
            assert dense_low is False
        else:
            self.low_forward = False
            assert dense_low is True
        if self.args.save:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.log_dir = 'runs/hier/point/FixS_' + current_time + "_HiDim_" + str(args.hi_dim) + \
                            '_HiRand_' + str(args.hi_random_eps) + "_NoLow_" + str(self.not_train_low) + \
                            "_HiLr_" + str(args.hi_lr_actor) + "_noise"
            self.writer = SummaryWriter(log_dir=self.log_dir)
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
                # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name + "_" + current_time)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
        # init hierarchical network
        self.real_goal_dim = 2
        self.init_network()

        # her sampler
        self.c = self.args.c  # interval of high level action
        self.env_params['max_high_steps'] = int(self.env_params['max_timesteps'] / self.c)
        self.hi_her_module = her_sampler(self.args.hi_replay_strategy, self.args.replay_k, self.args.distance,
                                          self.real_goal_dim, int(self.env_params['max_high_steps'] / 2),
                                         dense_reward=dense_high)
        self.low_her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.args.distance,
                                      self.real_goal_dim, self.args.future_step, dense_reward=dense_low)
        self.env_params['real_goal_dim'] = self.real_goal_dim
        self.hi_buffer = replay_buffer(self.env_params, self.args.buffer_size, self.hi_her_module.sample_her_transitions, 'max_high_steps')
        if args.env_name[:5] == "Fetch":
            self.low_buffer = replay_buffer_energy(self.env_params, self.args.buffer_size,
                                               self.low_her_module.sample_her_energy, args.env_name)
            if args.ebp:
                self.hi_buffer = replay_buffer_energy(self.env_params, self.args.buffer_size,
                                               self.hi_her_module.sample_her_energy, args.env_name, 'max_high_steps')
        else:
            self.low_buffer = replay_buffer(self.env_params, self.args.buffer_size, self.low_her_module.sample_her_transitions)

        not_load_buffer, not_load_high = self.not_train_low, self.not_train_low
        if self.resume == True:
            self.start_epoch = self.resume_epoch
            if not not_load_high:
                self.hi_actor_network.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/hi_actor_model.pt', map_location='cuda:1')[0])
                self.hi_critic_network.load_state_dict(torch.load(self.args.resume_path + \
                                                               '/hi_critic_model.pt', map_location='cuda:1')[0])
            self.low_actor_network.load_state_dict(torch.load(self.args.resume_path + \
                                                             '/low_actor_model.pt', map_location='cuda:1')[0])
            self.low_critic_network.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/low_critic_model.pt', map_location='cuda:1')[0])
            if not not_load_buffer:
                self.hi_buffer = torch.load(self.args.resume_path + '/hi_buffer.pt', map_location='cuda:1')
                self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')

        # sync target network
        self.sync_target()

        if hasattr(self.env.env, 'env'):
            self.animate = self.env.env.env.visualize_goal
        else:
            self.animate = self.args.animate
        self.distance_threshold = self.args.distance

        if args.contrastive_phi:
            self.representation = RepresentationNetwork(env_params, layer=3).to(args.device)
            self.representation_optim = torch.optim.Adam(self.representation.parameters(), lr=0.0001)
            if self.resume is True:
                self.representation.load_state_dict(torch.load(self.args.resume_path + \
                                                               '/phi_model.pt', map_location='cuda:1')[0])

    def adjust_lr_actor(self, epoch):
        lr_actor = self.args.lr_actor * (0.5 ** (epoch // self.args.lr_decay_actor))
        hi_lr_actor = self.args.hi_lr_actor * (0.5 ** (epoch // self.args.lr_decay_actor))
        for param_group in self.hi_actor_optim.param_groups:
            param_group['lr'] = hi_lr_actor
        for param_group in self.low_actor_optim.param_groups:
            param_group['lr'] = lr_actor

    def adjust_lr_critic(self, epoch):
        lr_critic = self.args.lr_critic * (0.5 ** (epoch // self.args.lr_decay_critic))
        for param_group in self.hi_critic_optim.param_groups:
            param_group['lr'] = lr_critic
        for param_group in self.low_critic_optim.param_groups:
            param_group['lr'] = lr_critic

    def learn(self):
        for epoch in range(self.start_epoch, self.args.n_epochs):
            if epoch > 0 and epoch % self.args.lr_decay_actor == 0:
                self.adjust_lr_actor(epoch)
            if epoch > 0 and epoch % self.args.lr_decay_critic == 0:
                self.adjust_lr_critic(epoch)

            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            hi_ep_obs, hi_ep_ag, hi_ep_g, hi_ep_actions = [], [], [], []
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']

            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    if t % self.c == 0:
                        final = bool(self.env_params['max_timesteps'] - t <= self.c)
                        if final or self.not_train_high:
                            hi_action = g.copy()[:self.real_goal_dim]
                        else:
                            hi_action = self.hi_policy(act_obs[:, :self.hi_dim], act_g, ag)
                            # print("ag, hi_action", ag, hi_action, np.linalg.norm(ag-hi_action))
                        # append high-level rollouts
                        hi_ep_obs.append(obs[:self.hi_dim].copy())
                        hi_ep_ag.append(ag.copy())
                        hi_ep_g.append(g.copy())
                        hi_ep_actions.append(hi_action.copy())
                        hi_action_tensor = torch.tensor(hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                    if self.not_train_low:
                        action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                    else:
                        action = self.explore_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                # feed the actions into the environment
                observation_new, _, _, info = self.env.step(action)
                if self.animate:
                    self.env.render()
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                # append rollouts
                ep_obs.append(obs[:self.low_dim].copy())
                ep_ag.append(ag.copy()[:self.real_goal_dim])
                ep_g.append(hi_action.copy())
                ep_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new
            ep_obs.append(obs[:self.low_dim].copy())
            ep_ag.append(ag.copy()[:self.real_goal_dim])
            hi_ep_obs.append(obs[:self.hi_dim].copy())
            hi_ep_ag.append(ag.copy())

            mb_obs = np.array([ep_obs])
            mb_ag = np.array([ep_ag])
            mb_g = np.array([ep_g])
            mb_actions = np.array([ep_actions])
            hi_mb_obs = np.array([hi_ep_obs])
            hi_mb_ag = np.array([hi_ep_ag])
            hi_mb_g = np.array([hi_ep_g])
            hi_mb_actions = np.array([hi_ep_actions])
            self.low_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
            self.hi_buffer.store_episode([hi_mb_obs, hi_mb_ag, hi_mb_g, hi_mb_actions])
            for n_batch in range(self.args.n_batches):
                if not self.not_train_low:
                    self._update_network(epoch, self.low_buffer, self.low_actor_target_network,
                                         self.low_critic_target_network,
                                         self.low_actor_network, self.low_critic_network, 'max_timesteps',
                                         self.low_actor_optim, self.low_critic_optim, use_forward_loss=self.low_forward, clip=self.low_use_clip)
                if n_batch % self.args.period == 0:
                    self._soft_update_target_network(self.low_actor_target_network, self.low_actor_network)
                    self._soft_update_target_network(self.low_critic_target_network, self.low_critic_network)
            for n_batch in range(self.args.hi_n_batches):
                if not self.not_train_high:
                    self._update_network(epoch, self.hi_buffer, self.hi_actor_target_network,
                                         self.hi_critic_target_network,
                                         self.hi_actor_network, self.hi_critic_network, 'max_high_steps',
                                         self.hi_actor_optim, self.hi_critic_optim, use_forward_loss=self.hi_forward, clip=self.use_clip)
                if n_batch % self.args.period == 0:
                    self._soft_update_target_network(self.hi_actor_target_network, self.hi_actor_network)
                    self._soft_update_target_network(self.hi_critic_target_network, self.hi_critic_network)
            # update phi
            if self.args.contrastive_phi and epoch % self.args.eval_interval == 0 and epoch > 0:
                self.update_phi(epoch)
            # start to do the evaluation
            if epoch % self.args.eval_interval == 0 and epoch != 0:
                success_rate, hier_success_rate = self._eval_agent(), self._eval_hier_agent()
                print('[{}] epoch is: {}, eval success rate is: {:.3f}, eval hier success rate is: {:.3f}'.format(datetime.now(), epoch,
                                                                               success_rate, hier_success_rate))
                if self.args.save:
                    print("log_dir: ", self.log_dir)
                    torch.save([self.hi_critic_network.state_dict()], self.model_path + '/hi_critic_model.pt')
                    torch.save([self.hi_actor_network.state_dict()], self.model_path + '/hi_actor_model.pt')
                    torch.save([self.low_critic_network.state_dict()], self.model_path + '/low_critic_model.pt')
                    torch.save([self.low_actor_network.state_dict()], self.model_path + '/low_actor_model.pt')
                    torch.save(self.hi_buffer, self.model_path + '/hi_buffer.pt')
                    torch.save(self.low_buffer, self.model_path + '/low_buffer.pt')
                    self.writer.add_scalar('Success_rate/low_' + self.args.env_name + self.args.metric, success_rate, epoch)
                    self.writer.add_scalar('Success_rate/hier_' + self.args.env_name + self.args.metric, hier_success_rate, epoch)
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

    def hi_policy(self, obs, goal, ag):
        print("hi_obs", obs)
        # random actions...
        if np.random.rand() < self.hi_random_eps:
            if self.args.env_name[:5] == "Fetch":
                hi_action = ag + np.random.randn(*ag.shape) * self.env.env.target_range / 2
                hi_action = hi_action[:self.real_goal_dim]
            else:
                hi_action = ag + np.random.randn(*ag.shape) * self.c / 4
        else:
            high_pi = self.hi_actor_network(obs, goal)
            hi_action = high_pi.cpu().numpy().squeeze()
            # add the gaussian
            hi_action += self.hi_noise_eps * self.maze_high * np.random.randn(*hi_action.shape)
        hi_action = np.clip(hi_action, self.hi_range[0], self.hi_range[1])
        return hi_action

    def random_policy(self, obs, goal):
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        return random_actions

    def test_policy(self, obs, goal):
        pi = self.low_actor_network(obs, goal)
        # convert the actions
        actions = pi.detach().cpu().numpy().squeeze()
        return actions

    def hi_test_policy(self, obs, goal):
        pi = self.hi_actor_network(obs, goal)
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
        dist = np.linalg.norm(ag_next[:, :self.real_goal_dim] - g_next[:, :self.real_goal_dim], axis=1)
        not_done = (dist > self.distance_threshold).astype(np.int32).reshape(-1, 1)

        # transfer them into the tensor
        obs_cur = torch.tensor(obs_cur, dtype=torch.float32).to(self.device)
        g_cur = torch.tensor(g_cur, dtype=torch.float32).to(self.device)
        ag_cur = torch.tensor(ag, dtype=torch.float32).to(self.device)
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
        if use_forward_loss:
            forward_loss = critic(obs_cur, ag_next, actions_tensor).pow(2).mean()
            critic_loss += forward_loss
        # the actor loss
        actions_real = actor(obs_cur, g_cur)
        actor_loss = -critic(obs_cur, g_cur, actions_real).mean()
        if T == 'max_timesteps':
            actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        else:
            actor_loss += self.args.action_l2 * ((actions_real - ag_cur[:, :self.real_goal_dim]) / self.maze_high).pow(2).mean()
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
            g = observation['desired_goal'][:self.real_goal_dim]
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    actions = policy(act_obs[:, :self.low_dim], act_g)
                observation_new, _, done, info = self.env.step(actions)
                if self.animate:
                    self.env.render()
                obs = observation_new['observation']
                g = observation_new['desired_goal'][:self.real_goal_dim]
                if done:
                    per_success_rate.append(info['is_success'])
                    break
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        if self.args.eval:
            print("global success rate", global_success_rate)
        return global_success_rate

    def _eval_hier_agent(self, n_test_rollouts=100):
        total_success_rate = []
        if not self.args.eval:
            n_test_rollouts = self.args.n_test_rollouts
        for _ in range(n_test_rollouts):
            per_success_rate = []
            observation = self.test_env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for num in range(self.env_params['max_test_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    if num % self.c == 0:
                        final = bool(self.env_params['max_timesteps'] - num < self.c)
                        if final:
                            hi_action = act_g
                        else:
                            hi_action = self.hi_test_policy(act_obs[:, :self.hi_dim], act_g)
                        hi_action_tensor = torch.tensor(hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                    action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                observation_new, rew, done, info = self.test_env.step(action)
                if self.animate:
                    self.test_env.render()
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                if done:
                    per_success_rate.append(info['is_success'])
                    break
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        if self.args.eval:
            print("hier success rate", global_success_rate)
        return global_success_rate

    def init_network(self):
        if self.args.env_name == "Fetch":
            self.hi_actor_network = Hi_actor(self.env_params, self.real_goal_dim, self.maze_high, self.shallow, sigmoid=True).to(self.device)
            self.hi_actor_target_network = Hi_actor(self.env_params, self.real_goal_dim, self.maze_high, self.shallow, sigmoid=True).to(self.device)
        else:
            self.hi_actor_network = Hi_actor(self.env_params, self.real_goal_dim, self.maze_high, self.shallow).to(self.device)
            self.hi_actor_target_network = Hi_actor(self.env_params, self.real_goal_dim, self.maze_high, self.shallow).to(self.device)
        self.hi_critic_network = Hi_critic(self.env_params, self.args, self.real_goal_dim, self.maze_high).to(self.device)
        self.hi_critic_target_network = Hi_critic(self.env_params, self.args, self.real_goal_dim, self.maze_high).to(self.device)
        self.low_actor_network = actor(self.env_params, self.real_goal_dim).to(self.device)
        self.low_actor_target_network = actor(self.env_params, self.real_goal_dim).to(self.device)
        self.low_critic_network = criticWrapper(self.env_params, self.args, self.real_goal_dim).to(self.device)
        self.low_critic_target_network = criticWrapper(self.env_params, self.args, self.real_goal_dim).to(self.device)

        self.start_epoch = 0

        # create the optimizer
        self.hi_actor_optim = torch.optim.Adam(self.hi_actor_network.parameters(), lr=self.args.hi_lr_actor)
        self.hi_critic_optim = torch.optim.Adam(self.hi_critic_network.parameters(), lr=self.args.lr_critic)
        self.low_actor_optim = torch.optim.Adam(self.low_actor_network.parameters(), lr=self.args.lr_actor)
        self.low_critic_optim = torch.optim.Adam(self.low_critic_network.parameters(), lr=self.args.lr_critic)

    def sync_target(self):
        # load the weights into the target networks
        self.hi_actor_target_network.load_state_dict(self.hi_actor_network.state_dict())
        self.hi_critic_target_network.load_state_dict(self.hi_critic_network.state_dict())
        self.low_actor_target_network.load_state_dict(self.low_actor_network.state_dict())
        self.low_critic_target_network.load_state_dict(self.low_critic_network.state_dict())


    def update_phi(self, epoch, not_align=True):
        if not_align:
            train_data = self.new_collect()
        else:
            train_data = self.collect()
        shuffle_idx = torch.randperm(train_data.size()[1])
        train_data = train_data[:, shuffle_idx]
        n_batches = int(train_data.size()[1] / self.args.batch_size)
        for n_batch in range(n_batches):
            from_idx, to_idx = n_batch * self.args.batch_size, (n_batch + 1) * self.args.batch_size
            sample_data = train_data[:, from_idx:to_idx]
            obs, obs_next = self.representation(sample_data[0]), self.representation(sample_data[1])
            min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
            if not_align:
                hi_obs, hi_obs_next = self.representation(sample_data[2]), self.representation(sample_data[3])
                max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
            else:
                neg_obs = self.representation(sample_data[2])
                max_dist = torch.clamp(1 - (obs - neg_obs).pow(2).mean(dim=1), min=0.)
            representation_loss = (min_dist + max_dist).mean()
            self.representation_optim.zero_grad()
            representation_loss.backward()
            self.representation_optim.step()
            if self.args.save:
                self.writer.add_scalar('Loss/phi_loss' + self.args.metric, representation_loss, epoch)

    def collect(self):
        all_obs = self.low_buffer.get_all_data()['obs'].copy()
        obs = all_obs[:, :-1-self.c, :]
        obs_next = all_obs[:, 1:-self.c, :]
        neg_obs = all_obs[:, self.c:-1, :]
        last_shape = obs.shape[2]
        obs, obs_next, neg_obs = obs.reshape(-1, last_shape), obs_next.reshape(-1, last_shape), neg_obs.reshape(-1, last_shape)

        train_data = np.array([obs, obs_next, neg_obs])
        train_data = torch.Tensor(train_data).to(self.device)
        return train_data

    def new_collect(self):
        all_obs = self.low_buffer.get_all_data()['obs'].copy()
        obs = all_obs[:, :-1, :]
        obs_next = all_obs[:, 1:, :]
        last_shape = obs.shape[2]
        obs, obs_next = obs.reshape(-1, last_shape), obs_next.reshape(-1, last_shape)
        len_obs = obs.shape[0]

        hi_all_obs = self.hi_buffer.get_all_data()['obs'].copy()
        hi_obs = hi_all_obs[:, :-1, :]
        hi_obs_next = hi_all_obs[:, 1:, :]
        hi_obs, hi_obs_next = hi_obs.reshape(-1, last_shape), hi_obs_next.reshape(-1, last_shape)

        tile_num = int(len_obs / self.c)
        hi_obs = np.tile(hi_obs[-tile_num:], (self.c, 1))
        hi_obs_next = np.tile(hi_obs_next[-tile_num:], (self.c, 1))

        train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
        train_data = torch.Tensor(train_data).to(self.device)
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
