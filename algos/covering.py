import os
import sys

sys.path.append('../')
from datetime import datetime
from tensorboardX import SummaryWriter
from models.networks import *
from algos.replay_buffer import replay_buffer, replay_buffer_energy
from algos.her import her_sampler

from algos.sac.model import QNetwork_out
from algos.sac.replay_memory import ReplayMemory
import random


class covering_agent:
    def __init__(self, args, env, env_params, test_env):
        self.args = args
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.device = args.device
        self.resume = args.resume
        self.resume_epoch = args.resume_epoch

        self.low_dim = env_params['obs']
        self.env_params['low_dim'] = self.low_dim
        self.hi_dim = env_params['obs']
        print("hi_dim", self.hi_dim)
        self.early_stop = 8000
        self.not_train_low = False

        if args.save:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.log_dir = 'runs/covering/' + str(args.env_name) + "_" + current_time + \
                            "_Seed_" + str(args.seed) + "_Reward_" + str(args.low_reward_coeff) + "_Early_" + str(self.early_stop) + \
                            "_Image_" + str(args.image)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name + "_" + current_time)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

        # init low-level network
        self.real_goal_dim = 3  # option num
        self.init_network()
        # init high-level agent
        self.Q_network = QNetwork_out(self.hi_dim + env_params['goal'], self.real_goal_dim, args.hidden_size).to(self.device)
        self.targetQ_network = QNetwork_out(self.hi_dim + env_params['goal'], self.real_goal_dim, args.hidden_size).to(self.device)

        self.env_params['real_goal_dim'] = self.real_goal_dim
        self.hi_buffer = ReplayMemory(args.buffer_size)

        # her sampler
        self.low_her_module = her_sampler(args.replay_strategy, args.replay_k, args.distance, args.future_step,
                                          dense_reward=True, direction_reward=False, low_reward_coeff=args.low_reward_coeff)
        self.low_buffer = replay_buffer(self.env_params, self.args.buffer_size, self.low_her_module.sample_her_transitions)

        if self.resume is True:
            self.start_epoch = self.resume_epoch
            self.low_actor_network.load_state_dict(torch.load(self.args.resume_path + \
                                                             '/low_actor_model.pt', map_location='cuda:1')[0])
            self.low_critic_network.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/low_critic_model.pt', map_location='cuda:1')[0])
            self.hi_buffer = torch.load(self.args.resume_path + '/hi_buffer.pt', map_location='cuda:1')
            self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')

        # sync target network
        self.sync_target()
        # create the optimizer
        self.q_optim = torch.optim.Adam(self.Q_network.parameters(), lr=self.args.lr)

        if hasattr(self.env.env, 'env'):
            self.animate = self.env.env.env.visualize_goal
        else:
            self.animate = self.args.animate

        self.phi_list, self.phi_optim_lst, self.buffer_list = [], [], []
        self.terminate_threshold = [0, 0, 0]
        for _ in range(self.real_goal_dim):
            representation = RepresentationNetwork(env_params, 3, 50, 1).to(args.device)
            representation_optim = torch.optim.Adam(representation.parameters(), lr=0.0001)
            phi_buffer = ReplayMemory(args.buffer_size)
            self.phi_list.append(representation)
            self.phi_optim_lst.append(representation_optim)
            self.buffer_list.append(phi_buffer)


    def adjust_lr_actor(self, epoch):
        lr_actor = self.args.lr_actor * (0.5 ** (epoch // self.args.lr_decay_actor))
        for param_group in self.low_actor_optim.param_groups:
            param_group['lr'] = lr_actor

    def adjust_lr_critic(self, epoch):
        lr_critic = self.args.lr_critic * (0.5 ** (epoch // self.args.lr_decay_critic))
        for param_group in self.low_critic_optim.param_groups:
            param_group['lr'] = lr_critic

    def learn(self):
        # random collect to update phi
        for option in range(self.real_goal_dim):
            for epoch in range(4):
                observation = self.env.reset()
                obs = observation['observation']
                g = observation['desired_goal']
                for t in range(self.env_params['max_timesteps']):
                    action = self.random_policy(obs, g)
                    # feed the actions into the environment
                    observation_new, r, _, info = self.env.step(action)
                    obs_new = observation_new['observation']
                    self.buffer_list[option].push(obs, action, r, obs_new, 0, epoch)
                    obs = obs_new

                for _ in range(5):
                    self.slow_update_phi(option)
            self.update_terminate(option)

        # online learning
        for epoch in range(self.start_epoch, self.args.n_epochs):
            if epoch > 0 and epoch % self.args.lr_decay_actor == 0:
                self.adjust_lr_actor(epoch)
            if epoch > 0 and epoch % self.args.lr_decay_critic == 0:
                self.adjust_lr_critic(epoch)

            if epoch > self.early_stop:
                self.not_train_low = True

            ep_obs, ep_ag, ep_g, ep_actions, ep_done = [], [], [], [], []
            last_hi_obs = None
            success = 0
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            terminate = True

            for t in range(self.env_params['max_timesteps']):
                act_obs, act_g = self._preproc_inputs(obs, g)
                # if terminate, select hi_action again
                if terminate:
                    hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
                    # append high-level rollouts
                    if last_hi_obs is not None:
                        mask = float(not done)
                        self.hi_buffer.push(last_hi_obs, last_hi_a, last_hi_r, hi_act_obs, mask, epoch)
                    hi_action, option = self.hi_explore(hi_act_obs)
                    last_hi_obs = hi_act_obs.copy()
                    last_hi_a = hi_action.copy()
                    last_hi_r = 0.
                    done = False
                    hi_action_tensor = torch.tensor(hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                    # update high-level policy
                    if len(self.hi_buffer) > self.args.batch_size:
                        self.update_hi()
                        self._soft_update_target_network(self.targetQ_network, self.Q_network)
                with torch.no_grad():
                    if self.not_train_low:
                        action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                    else:
                        action = self.explore_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                # feed the actions into the environment
                observation_new, r, _, info = self.env.step(action)
                if info['is_success']:
                    done = True

                ag = self.phi_list[option](act_obs).detach().cpu().numpy()[0]
                obs_new = observation_new['observation']
                act_obs_new, act_g = self._preproc_inputs(obs_new, g)
                ag_new = self.phi_list[option](act_obs_new).detach().cpu().numpy()[0]

                # save phi buffer
                self.buffer_list[option].push(obs, action, r, obs_new, 0, epoch)
                if done is False:
                    last_hi_r += r

                # append rollouts
                ep_obs.append(obs[:self.low_dim].copy())
                ep_ag.append(ag.copy())
                ep_g.append(hi_action.copy())
                ep_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new
                if ag < self.terminate_threshold[option]:
                    terminate = True
                else:
                    terminate = False
                ep_done.append(np.array([terminate]))

            # slowly update phi
            if not self.not_train_low:
                for option in range(self.real_goal_dim):
                    for _ in range(2):
                        self.slow_update_phi(option)
                    self.update_terminate(option)


            ep_obs.append(obs[:self.low_dim].copy())
            ep_ag.append(ag.copy())
            mask = float(not done)
            hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
            self.hi_buffer.push(last_hi_obs, last_hi_a, last_hi_r, hi_act_obs, mask, epoch)

            mb_obs = np.array([ep_obs])
            mb_ag = np.array([ep_ag])
            mb_g = np.array([ep_g])
            mb_actions = np.array([ep_actions])
            mb_done = np.array([ep_done])
            self.low_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, success, mb_done])

            # update low-level
            if not self.not_train_low:
                for n_batch in range(self.args.n_batches):
                    self._update_network(epoch, self.low_buffer, self.low_actor_target_network,
                                         self.low_critic_target_network,
                                         self.low_actor_network, self.low_critic_network,
                                         self.low_actor_optim, self.low_critic_optim)
                    if n_batch % self.args.period == 0:
                        self._soft_update_target_network(self.low_actor_target_network, self.low_actor_network)
                        self._soft_update_target_network(self.low_critic_target_network, self.low_critic_network)

            # start to do the evaluation
            if epoch % self.args.eval_interval == 0 and epoch != 0:
                farthest_success_rate = self._eval_hier_agent(env=self.test_env)
                random_success_rate = self._eval_hier_agent(env=self.env)
                print('[{}] epoch is: {}, eval hier success rate is: {:.3f}'.format(datetime.now(), epoch, random_success_rate))
                if self.args.save:
                    print("log_dir: ", self.log_dir)
                    torch.save([self.low_critic_network.state_dict()], self.model_path + '/low_critic_model.pt')
                    torch.save([self.low_actor_network.state_dict()], self.model_path + '/low_actor_model.pt')
                    torch.save(self.hi_buffer, self.model_path + '/hi_buffer.pt')
                    torch.save(self.low_buffer, self.model_path + '/low_buffer.pt')
                    self.writer.add_scalar('Success_rate/hier_farthest_' + self.args.env_name, farthest_success_rate, epoch)
                    self.writer.add_scalar('Success_rate/hier_random_' + self.args.env_name, random_success_rate, epoch)

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
    def _update_network(self, epoch, buffer, actor_target, critic_target, actor, critic, actor_optim, critic_optim):
        # sample the episodes
        transitions = buffer.sample(self.args.batch_size)
        ag_next = transitions['ag_next']

        # start to do the update
        obs_cur = transitions['obs']
        g_cur = transitions['g']
        obs_next = transitions['obs_next']
        g_next = transitions['g']

        # done
        done = transitions['done']
        not_done = (done == 0)
        reward = (transitions['ag'] - ag_next) * self.args.low_reward_coeff

        # transfer them into the tensor
        obs_cur = torch.tensor(obs_cur, dtype=torch.float32).to(self.device)
        g_cur = torch.tensor(g_cur, dtype=torch.float32).to(self.device)
        obs_next = torch.tensor(obs_next, dtype=torch.float32).to(self.device)
        g_next = torch.tensor(g_next, dtype=torch.float32).to(self.device)
        not_done = torch.tensor(not_done, dtype=torch.int32).to(self.device)

        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32).to(self.device)
        r_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device)

        # calculate the target Q value function
        with torch.no_grad():
            actions_next = actor_target(obs_next, g_next)
            q_next_value = critic_target(obs_next, g_next, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + critic_target.gamma * q_next_value * not_done
            target_q_value = target_q_value.detach()

        # the q loss
        real_q_value = critic(obs_cur, g_cur, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

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
            name = 'low'
            self.writer.add_scalar('Loss/' + name + '_actor_loss' + self.args.metric, actor_loss, epoch)
            self.writer.add_scalar('Loss/' + name + '_critic_loss' + self.args.metric, critic_loss, epoch)

    def _eval_hier_agent(self, env, n_test_rollouts=100):
        total_success_rate = []
        if not self.args.eval:
            n_test_rollouts = self.args.n_test_rollouts

        for roll in range(n_test_rollouts):
            per_success_rate = []
            observation = env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            terminate = True
            for num in range(self.env_params['max_test_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    if terminate:
                        hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
                        hi_action, option = self.hi_explore(hi_act_obs, test=True)
                        hi_action_tensor = torch.tensor(hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                    action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                observation_new, rew, done, info = env.step(action)
                obs = observation_new['observation']
                act_obs, act_g = self._preproc_inputs(obs, g)
                ag = self.phi_list[option](act_obs).detach().cpu().numpy()[0]
                if ag < self.terminate_threshold[option]:
                    terminate = True
                else:
                    terminate = False
                if done:
                    per_success_rate.append(info['is_success'])
                    break
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])

        return global_success_rate

    def init_network(self):
        self.low_actor_network = actor(self.env_params, self.real_goal_dim).to(self.device)
        self.low_actor_target_network = actor(self.env_params, self.real_goal_dim).to(self.device)
        self.low_critic_network = criticWrapper(self.env_params, self.args, self.real_goal_dim).to(self.device)
        self.low_critic_target_network = criticWrapper(self.env_params, self.args, self.real_goal_dim).to(self.device)

        self.start_epoch = self.real_goal_dim * 4

        # create the optimizer
        self.low_actor_optim = torch.optim.Adam(self.low_actor_network.parameters(), lr=self.args.lr_actor)
        self.low_critic_optim = torch.optim.Adam(self.low_critic_network.parameters(), lr=self.args.lr_critic)

    def sync_target(self):
        # load the weights into the target networks
        self.low_actor_target_network.load_state_dict(self.low_actor_network.state_dict())
        self.low_critic_target_network.load_state_dict(self.low_critic_network.state_dict())
        # high-level
        self.targetQ_network.load_state_dict(self.Q_network.state_dict())

    def slow_update_phi(self, option):
        sample_data = self.slow_collect(option)
        sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)

        obs, obs_next = self.phi_list[option](sample_data[0]), self.phi_list[option](sample_data[1])
        min_dist = (obs - obs_next).pow(2)
        other_obs = self.phi_list[option](sample_data[2])
        max_dist = (obs.pow(2) - 1) * (other_obs.pow(2) - 1) + obs.pow(2) * other_obs.pow(2)
        representation_loss = (min_dist + max_dist).mean()

        self.phi_optim_lst[option].zero_grad()
        representation_loss.backward()
        self.phi_optim_lst[option].step()

    def slow_collect(self, option, batch_size=100):
        hi_obs, _, _, hi_obs_next, _ = self.buffer_list[option].sample(batch_size)
        other_obs, _, _, _, _ = self.buffer_list[option].sample(batch_size)
        hi_obs, hi_obs_next = hi_obs[:, :self.env_params['obs']], hi_obs_next[:, :self.env_params['obs']]
        train_data = np.array([hi_obs, hi_obs_next, other_obs])
        return train_data


    def hi_explore(self, obs, test=False):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        candidate = []
        for option in range(self.real_goal_dim):
            f_value = self.phi_list[option](obs[:, :self.hi_dim]).detach().cpu().numpy()
            if f_value > self.terminate_threshold[option]:
                candidate.append(option)
        if candidate == []:
            best_action = np.random.randint(self.real_goal_dim)
        else:
            q = self.Q_network(obs)[0][candidate]
            max_index = torch.argmax(q)
            best_action = candidate[max_index]
            if not test:
                if random.random() < self.args.random_eps:
                    best_action = np.random.randint(self.real_goal_dim)

        action_out = np.zeros(self.real_goal_dim)
        action_out[best_action] = 1
        return action_out, best_action

    def update_terminate(self, option):
        batch_size = 100
        other_obs, _, _, _, _ = self.buffer_list[option].sample(batch_size)
        sample_data = torch.tensor(other_obs, dtype=torch.float32).to(self.device)
        other_obs = self.phi_list[option](sample_data)
        sorted, _ = torch.sort(other_obs)
        min_k = sorted[int(batch_size * 0.1)][0].detach().cpu().numpy()
        self.terminate_threshold[option] = min_k

    def update_hi(self):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.hi_buffer.sample(batch_size=self.args.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            qf1_next_target = self.targetQ_network(next_state_batch)
            # double DQN
            qf1_next_current = self.Q_network(next_state_batch)
            a_index = torch.argmax(qf1_next_current, dim=1)
            index = torch.arange(self.args.batch_size, dtype=int)
            next_Q = qf1_next_target[index, a_index]
            next_Q = torch.reshape(next_Q, (-1, 1))
            next_q_value = reward_batch + mask_batch * self.args.gamma * next_Q

        qf1 = self.Q_network(state_batch)
        action = action_batch.argmax(dim=1)
        index = torch.arange(self.args.batch_size, dtype=int)
        qf1 = qf1[index, action]
        qf1 = torch.reshape(qf1, (-1, 1))

        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        self.q_optim.zero_grad()
        qf1_loss.backward()
        self.q_optim.step()































