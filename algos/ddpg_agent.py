import os
import sys

sys.path.append('../')
from datetime import datetime
from tensorboardX import SummaryWriter
from models.networks import *
from algos.replay_buffer import replay_buffer, replay_buffer_energy
from algos.her import her_sampler
from planner.vis_pointmaze import vis_pointmaze
from planner.goal_plan import *
import matplotlib.pyplot as plt
import torch.nn.functional as F

# flat ddpg agent
class ddpg_agent:
    def __init__(self, args, env, env_params, test_env):
        self.args = args
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.device = args.device
        self.resume = args.resume
        self.resume_epoch = args.resume_epoch
        if self.args.save:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.log_dir = 'runs/flat/' + str(args.env_name) + '/Seed_' + str(args.seed) + current_time + \
                           '_Image_' + str(args.image) + '_ActionL2_' + str(args.action_l2) + '_Period_' + str(args.period) + \
                           '_Gamma_' + str(args.gamma)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
                # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name + "_" + current_time)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
        env_params['low_dim'] = env_params['obs']
        print("obs dim", env_params['obs'])
        env_params['real_goal_dim'] = env_params['goal']
        self.actor_network = actor(env_params, env_params['goal'])
        self.actor_target_network = actor(env_params, env_params['goal'])
        self.critic_network = criticWrapper(env_params, self.args, env_params['goal'])
        self.critic_target_network = criticWrapper(env_params, self.args, env_params['goal'])

        self.start_epoch = 0
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        self.actor_network.to(self.device)
        self.critic_network.to(self.device)
        self.actor_target_network.to(self.device)
        self.critic_target_network.to(self.device)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # her sampler
        self.real_goal_dim = 2
        self.her_module = her_sampler(args.replay_strategy, args.replay_k, args.distance,
                                      args.future_step, True, False, 0.1, low_idxs=0)
        if args.env_name[:5] == "Fetch":
            # use ebp in fetch env
            self.buffer = replay_buffer_energy(self.env_params, self.args.buffer_size,
                                               self.her_module.sample_her_energy, args.env_name)
        else:
            self.buffer = replay_buffer(env_params, args.buffer_size, self.her_module.sample_her_transitions)

        if self.resume == True:
            self.start_epoch = self.resume_epoch
            self.actor_network.load_state_dict(torch.load(self.args.resume_path + \
                                                          '/actor_model.pt', map_location='cuda:1')[0])
            self.critic_network.load_state_dict(torch.load(self.args.resume_path + \
                                                           '/critic_model.pt', map_location='cuda:1')[0])
            self.buffer = torch.load(self.args.resume_path + '/replaybuffer.pt', map_location='cuda:1')

        if hasattr(self.env.env, 'env'):
            self.animate = self.env.env.env.visualize_goal
        else:
            self.animate = self.args.animate
        self.distance_threshold = self.args.distance


    def adjust_lr_actor(self, epoch):
        lr_actor = self.args.lr_actor * (0.5 ** (epoch // self.args.lr_decay_actor))
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr_actor

    def adjust_lr_critic(self, epoch):
        lr_critic = self.args.lr_critic * (0.5 ** (epoch // self.args.lr_decay_critic))
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr_critic

    def learn(self):
        for epoch in range(self.start_epoch, self.args.n_epochs):
            if epoch > 0 and epoch % self.args.lr_decay_actor == 0:
                self.adjust_lr_actor(epoch)
            if epoch > 0 and epoch % self.args.lr_decay_critic == 0:
                self.adjust_lr_critic(epoch)

            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']

            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    action = self.explore_policy(act_obs, act_g)
                    # feed the actions into the environment
                observation_new, _, _, info = self.env.step(action)
                if self.animate:
                    self.env.render()
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())

            if self.args.save and self.args.env_name == "NChain-v1":
                self.writer.add_scalar('Explore/coverage_' + self.args.env_name, self.env.env.coverage,
                                       epoch)
            # print("coverage", self.env.env.coverage)

            mb_obs = np.array([ep_obs])
            mb_ag = np.array([ep_ag])
            mb_g = np.array([ep_g])
            mb_actions = np.array([ep_actions])
            self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, 0, 0, mb_ag])
            # self.store_figure(epoch)
            for n_batch in range(self.args.n_batches):
                self._update_network(epoch)
                if n_batch % self.args.period == 0:
                    self._soft_update_target_network(self.actor_target_network, self.actor_network)
                    self._soft_update_target_network(self.critic_target_network, self.critic_network)

            if epoch % self.args.eval_interval == 0 and epoch != 0:
                success_rate = self._eval_agent()
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch,
                                                                               success_rate))
                if self.args.save:
                    print("log_dir: ", self.log_dir)
                    torch.save([self.critic_network.state_dict()], \
                               self.model_path + '/critic_model.pt')
                    torch.save([self.actor_network.state_dict()], \
                               self.model_path + '/actor_model_{}.pt'.format(epoch))
                    torch.save(self.buffer, self.model_path + '/replaybuffer.pt')
                    self.writer.add_scalar('Success_rate/flat_' + self.args.env_name + self.args.metric, success_rate, epoch)

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
        pi = self.actor_network(obs, goal)
        action = self._select_actions(pi)
        return action

    def random_policy(self, obs, goal):
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        return random_actions

    def test_policy(self, obs, goal):
        pi = self.actor_network(obs, goal)
        # convert the actions
        actions = pi.detach().cpu().numpy().squeeze()
        return actions

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self, epoch):
        # sample the episodes
        transitions, _ = self.buffer.sample(self.args.batch_size)
        # print("transitions", transitions)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
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
        obs_next = torch.tensor(obs_next, dtype=torch.float32).to(self.device)
        g_next = torch.tensor(g_next, dtype=torch.float32).to(self.device)
        ag_next = torch.tensor(ag_next, dtype=torch.float32).to(self.device)
        not_done = torch.tensor(not_done, dtype=torch.int32).to(self.device)

        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32).to(self.device)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32).to(self.device)
        # calculate the target Q value function
        with torch.no_grad():
            actions_next = self.actor_target_network(obs_next, g_next)
            q_next_value = self.critic_target_network(obs_next, g_next, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.critic_target_network.gamma * q_next_value * not_done
            target_q_value = target_q_value.detach()
            # clip_return = 1 / (1 - self.critic_target_network.gamma)
            clip_return = self.env_params['max_timesteps']
            target_q_value = torch.clamp(target_q_value, -clip_return, 0.)
        # the q loss
        real_q_value = self.critic_network(obs_cur, g_cur, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # forward_loss = self.critic_network(obs_cur, ag_next, actions_tensor).pow(2).mean()
        # critic_loss += forward_loss
        # the actor loss
        actions_real = self.actor_network(obs_cur, g_cur)
        actor_loss = -self.critic_network(obs_cur, g_cur, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 1)
        self.critic_optim.step()

        if self.args.save:
            self.writer.add_scalar('Loss/actor_loss' + self.args.metric, actor_loss, epoch)
            self.writer.add_scalar('Loss/critic_loss' + self.args.metric, critic_loss, epoch)

    # do the evaluation
    def _eval_agent(self, policy=None, n_test_rollouts=10):
        if policy is None:
            policy = self.test_policy

        total_success_rate = []
        if not self.animate:
            n_test_rollouts = self.args.n_test_rollouts
        for _ in range(n_test_rollouts):
            per_success_rate = []
            observation = self.test_env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    actions = policy(act_obs, act_g)
                observation_new, _, done, info = self.test_env.step(actions)
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
            print("global success rate", global_success_rate)
        return global_success_rate

    def multi_eval(self):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = 'runs/flat/' + str(self.args.env_name) + '/' + current_time + \
                       "_Image_" + str(self.args.image) + "_Resolution_10_" + \
                       "_Seed_" + str(self.args.seed) + "_Eval_" + str(self.args.eval)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        for i in range(50, 8050, 50):
            try:
                self.actor_network.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/actor_model_{}.pt'.format(i), map_location='cuda:1')[0])
            except:
                print("Epoch:", i, " No such file !!!")

            eval_success = self._eval_agent()
            self.writer.add_scalar('Success_rate/hier_farthest_' + self.args.env_name, eval_success, i)
