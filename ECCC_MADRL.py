import time

import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
import numpy as np
import random
from copy import deepcopy
from numpy import savetxt
from numpy import loadtxt

from mec_env import N_SERVERS, K_CHANNELS, S_ES, CAPABILITY_ES
from utils import to_tensor_var
from Model import ActorNetwork, CriticNetwork
from prioritized_memory import Memory
from mec_env import ENV_MODE, N_UNITS


class ECCC_MADDPG(object):
    def __init__(self, InfdexofResult, env, env_eval, n_agents, state_dim, action_dim, action_lower_bound,
                 action_higher_bound,
                 memory_capacity=10000, target_tau=0.001, reward_gamma=0.99, reward_scale=0.01, done_penalty=None,#reward_scale=1.
                 actor_output_activation=torch.tanh, actor_lr=0.0001, critic_lr=0.001,
                 optimizer_type="adam", max_grad_norm=None, batch_size=64, episodes_before_train=64,
                 epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=None, use_cuda=False):
        self.n_agents = n_agents
        self.env = env
        self.env_eval = env_eval
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lower_bound = [0] * (N_SERVERS + 1) + [0.01, 0.01]
        self.action_higher_bound = [1] * (N_SERVERS + 1) + [1, 1]

        self.env_state = env.reset_mec()
        self.n_episodes = 0
        self.roll_out_n_steps = 1

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.memory = Memory(memory_capacity)
        self.actor_output_activation = actor_output_activation
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        if epsilon_decay == None:
            print("epsilon_decay is NOne")
            exit()
        else:
            self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.target_tau = target_tau

        # 客户端代理网络 - 每个设备一个
        self.actors = [ActorNetwork(self.state_dim, self.action_dim, self.actor_output_activation)] * self.n_agents
        
        # 主控代理网络 - 全局Critic
        critic_state_dim = self.n_agents * self.state_dim
        critic_action_dim = self.n_agents * self.action_dim
        self.critics = [CriticNetwork(critic_state_dim, critic_action_dim, self.state_dim, self.action_dim)] * 1

        # to ensure target network and learning network has the same weights
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        if optimizer_type == "adam":
            self.actors_optimizer = [Adam(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [Adam(c.parameters(), lr=self.critic_lr) for c in self.critics]
        elif optimizer_type == "rmsprop":
            self.actors_optimizer = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [RMSprop(c.parameters(), lr=self.critic_lr) for c in self.critics]

        if self.use_cuda:
            for i in range(self.n_agents):
                self.actors[i].cuda()
                self.critics[i].cuda()
                self.actors_target[i].cuda()
                self.critics_target[i].cuda()

        self.eval_episode_rewards = []
        self.server_episode_constraint_exceeds = []
        self.energy_episode_constraint_exceeds = []
        self.time_episode_constraint_exceeds = []
        self.eval_step_rewards = []
        self.mean_rewards = []

        self.episodes = []
        self.Training_episodes = []

        self.Training_episode_rewards = []
        self.Training_step_rewards = []

        self.InfdexofResult = InfdexofResult
        # self.save_models('./checkpoint/models_checkpoint'+str(self.InfdexofResult)+'.pth')
        self.results = []
        self.Training_results = []
        self.serverconstraints = []
        self.energyconstraints = []
        self.timeconstraints = []

    def interact(self, MAX_EPISODES, EPISODES_BEFORE_TRAIN, NUMBER_OF_EVAL_EPISODES):

        for episode in range(MAX_EPISODES):
            if episode % 50 == 0:  # 每50轮打印一次
                print(f"[训练进度] Episode {episode}/{MAX_EPISODES}")
                print(
                    f"\n[汇总] Episode {self.n_episodes}\n"
                    f"评估平均奖励: {np.mean(self.mean_rewards[-5:]):.2f}\n"
                )
        while self.n_episodes < MAX_EPISODES:
            self.env_state = self.env.reset_mec()
            if self.n_episodes >= EPISODES_BEFORE_TRAIN:
                self.evaluate(NUMBER_OF_EVAL_EPISODES)
                self.evaluateAtTraining(NUMBER_OF_EVAL_EPISODES)
            # EVAL_INTERVAL = 1  # 新增：每1轮评估一次
            # if self.n_episodes >= EPISODES_BEFORE_TRAIN and self.n_episodes % EVAL_INTERVAL == 0:
            #     self.evaluate(NUMBER_OF_EVAL_EPISODES)
            #     self.evaluateAtTraining(NUMBER_OF_EVAL_EPISODES)
            self.agent_rewards = [[] for n in range(self.n_agents)]
            done = False
            while not done:
                state = self.env_state
                actor_action, critic_action, hybrid_action = self.choose_action(state, False)
                next_state, reward, done, _, _ = self.env.step_mec(hybrid_action)
                self.Training_step_rewards.append(np.mean(reward))
                if done:
                    self.Training_episode_rewards.append(np.sum(np.array(self.Training_step_rewards)))
                    self.Training_step_rewards = []
                    if self.done_penalty is not None:
                        reward = self.done_penalty
                    self.n_episodes += 1
                else:
                    self.env_state = next_state
                self.append_sample(state, actor_action, critic_action, reward, next_state, done)
            if self.n_episodes >= EPISODES_BEFORE_TRAIN:
                self.train()
                pass

        print("[训练完成] 所有轮次训练结束！")  # 新增

    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def append_sample(self, states, actor_actions, critic_actions, rewards, next_states, dones):
        error = 0
        target_q = 0
        current_q = 0
        states_var = to_tensor_var(states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actor_actions_var = to_tensor_var(actor_actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        critic_actions_var = to_tensor_var(critic_actions, self.use_cuda).view(-1, self.n_agents, 1)
        rewards_var = to_tensor_var(rewards, self.use_cuda).view(-1, self.n_agents, 1)
        next_states_var = to_tensor_var(next_states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        whole_states_var = states_var.view(-1, self.n_agents * self.state_dim)
        whole_actor_actions_var = actor_actions_var.view(-1, self.n_agents * self.action_dim)
        whole_next_states_var = next_states_var.view(-1, self.n_agents * self.state_dim)
        # dones_var = to_tensor_var(dones, self.use_cuda).view(-1, 1)
        nextactor_actions = []
        # Calculate next target actions for each agent
        for agent_id in range(self.n_agents):
            next_action_var = self.actors_target[agent_id](next_states_var[:, agent_id, :])
            if self.use_cuda:
                nextactor_actions.append(next_action_var.data.cpu())
            else:
                nextactor_actions.append(next_action_var.data)
        # Concatenate the next target actions into a single tensor
        nextactor_actions_var = torch.cat(nextactor_actions, dim=1)
        nextactor_actions_var = nextactor_actions_var.view(-1, actor_actions_var.size(1), actor_actions_var.size(2))
        whole_nextactor_actions_var = nextactor_actions_var.view(-1, self.n_agents * self.action_dim)
        # target prediction
        nextperQs = []
        for nexta in range(self.n_agents):  # to find the maxQ
            if nextactor_actions_var[0, nexta, 0] >= 0:
                nextperQs.append(self.critics_target[0](whole_next_states_var[0], whole_nextactor_actions_var[0],
                                                        next_states_var[0, nexta, :],
                                                        nextactor_actions_var[0, nexta, :]).detach())
        if len(nextperQs) == 0:
            tar_perQ = self.critics_target[0](whole_next_states_var[0], whole_nextactor_actions_var[0],
                                              torch.zeros(self.state_dim), torch.zeros(self.action_dim)).detach()
        else:
            tar_perQ = max(nextperQs)
        tar = self.reward_scale * rewards_var[0, 0, :] + self.reward_gamma * tar_perQ * (1. - dones)
        cselected = 0
        for a in range(self.n_agents):
            if critic_actions_var[0, a, 0] == 1:
                # current prediction
                curr_perQ = self.critics[0](whole_states_var[0], whole_actor_actions_var[0], states_var[0, a, :],
                                            actor_actions_var[0, a, :]).detach()
                error += (tar - curr_perQ) ** 2
                cselected += 1
        if cselected == 0:  # if all tasks were allocated locally, the feedback should be sent using the commbined local decision and a fake perAgent that learns the best Q value for that situation
            curr_perQ = self.critics[0](whole_states_var[0], whole_actor_actions_var[0], torch.zeros(self.state_dim),
                                        torch.zeros(self.action_dim)).detach()
            error += (tar - curr_perQ) ** 2
        self.memory.addorupdate(error, (states, actor_actions, critic_actions, rewards, next_states, dones))

    # train on a sample batch
    def train(self):
        # do not train until exploration is enough
        if self.n_episodes <= self.episodes_before_train:
            return
        tryfetch = 0
        while tryfetch < 3:
            mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
            # print("idxs, is_weights", len(idxs), len(is_weights))
            mini_batch = np.array(mini_batch, dtype=object).transpose()
            if any(not isinstance(arr, np.ndarray) for arr in mini_batch[0]) or any(
                    not isinstance(arr, np.ndarray) for arr in mini_batch[2]):
                if tryfetch < 3:
                    tryfetch += 1
                else:
                    print("mini_batch = ", mini_batch)
                    exit()
            else:
                break
        errors = np.zeros(self.batch_size)
        states = np.vstack(mini_batch[0])
        actor_actions = np.vstack(mini_batch[1])
        critic_actions = np.vstack(mini_batch[2])
        rewards = np.vstack(mini_batch[3])
        next_states = np.vstack(mini_batch[4])
        dones = mini_batch[5]

        # bool to binary
        dones = dones.astype(int)
        states_var = to_tensor_var(states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actor_actions_var = to_tensor_var(actor_actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        critic_actions_var = to_tensor_var(critic_actions, self.use_cuda).view(-1, self.n_agents, 1)
        rewards_var = to_tensor_var(rewards, self.use_cuda).view(-1, self.n_agents, 1)
        next_states_var = to_tensor_var(next_states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        dones_var = to_tensor_var(dones, self.use_cuda).view(-1, 1)
        whole_states_var = states_var.view(-1, self.n_agents * self.state_dim)
        whole_actor_actions_var = actor_actions_var.view(-1, self.n_agents * self.action_dim)
        whole_next_states_var = next_states_var.view(-1, self.n_agents * self.state_dim)

        nextactor_actions = []
        # Calculate next target actions for each agent
        for agent_id in range(self.n_agents):
            next_action_var = self.actors_target[agent_id](next_states_var[:, agent_id, :])
            if self.use_cuda:
                nextactor_actions.append(next_action_var)
            else:
                nextactor_actions.append(next_action_var)
        # Concatenate the next target actions into a single tensor
        nextactor_actions_var = torch.cat(nextactor_actions, dim=1)
        nextactor_actions_var = nextactor_actions_var.view(-1, actor_actions_var.size(1), actor_actions_var.size(2))
        whole_nextactor_actions_var = nextactor_actions_var.view(-1, self.n_agents * self.action_dim)

        # common critic
        agent_id = 0
        target_q = []
        current_q = []
        for b in range(self.batch_size):
            # target prediction
            nextperQs = []
            for nexta in range(self.n_agents):  # to find the maxQ
                if nextactor_actions_var[b, nexta, 0] >= 0:
                    nextperQs.append(
                        self.critics_target[agent_id](whole_next_states_var[b], whole_nextactor_actions_var[b],
                                                      next_states_var[b, nexta, :], nextactor_actions_var[b, nexta, :]))
            if len(nextperQs) == 0:
                tar_perQ = self.critics_target[agent_id](whole_next_states_var[b], whole_nextactor_actions_var[b],
                                                         torch.zeros(self.state_dim), torch.zeros(self.action_dim))
            else:
                tar_perQ = max(nextperQs)
            tar = self.reward_scale * rewards_var[b, agent_id, :] + self.reward_gamma * tar_perQ * (1. - dones_var[b])
            cselected = 0
            for a in range(self.n_agents):
                if critic_actions_var[b, a, 0] == 1:
                    target_q.append(tar * is_weights[b])
                    # current prediction
                    curr_perQ = self.critics[agent_id](whole_states_var[b], whole_actor_actions_var[b],
                                                       states_var[b, a, :], actor_actions_var[b, a, :])
                    current_q.append(curr_perQ * is_weights[b])
                    errors[b] += (curr_perQ - tar) ** 2
                    cselected += 1
            if cselected == 0:  # if all tasks were allocated locally, the feedback should be sent using the commbined local decision and a fake perAgent that learns the best Q value for that situation
                target_q.append(tar * is_weights[b])
                curr_perQ = self.critics[agent_id](whole_states_var[b], whole_actor_actions_var[b],
                                                   torch.zeros(self.state_dim), torch.zeros(self.action_dim))
                current_q.append(curr_perQ * is_weights[b])
                errors[b] += (curr_perQ - tar) ** 2
        current_q = torch.stack(current_q, dim=0)
        target_q = torch.stack(target_q, dim=0)
        critic_loss = nn.MSELoss()(current_q, target_q)
        critic_loss.requires_grad_(True)
        self.critics_optimizer[agent_id].zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), self.max_grad_norm)
        self.critics_optimizer[agent_id].step()
        self._soft_update_target(self.critics_target[agent_id], self.critics[agent_id])  # update target
        # print("critic_loss",critic_loss.detach())

        # different actors
        for agent_id in range(self.n_agents):
            newactor_actions = []
            # Calculate new actions for each agent
            for agent_id in range(self.n_agents):
                newactor_action_var = self.actors[agent_id](states_var[:, agent_id, :])
                if self.use_cuda:
                    newactor_actions.append(
                        newactor_action_var)  # newactor_actions.append(newactor_action_var.data.cpu())
                else:
                    newactor_actions.append(newactor_action_var)  # newactor_actions.append(newactor_action_var.data)
            # Concatenate the new actions into a single tensor
            newactor_actions_var = torch.cat(newactor_actions, dim=1)
            newactor_actions_var = newactor_actions_var.view(-1, actor_actions_var.size(1), actor_actions_var.size(2))
            whole_newactor_actions_var = newactor_actions_var.view(-1, self.n_agents * self.action_dim)
            actor_loss = []
            for b in range(self.batch_size):
                Qselected = []
                for a in range(self.n_agents):
                    if newactor_actions_var[b, a, 0] >= 0:  # if it is delegated to the master agent
                        perQ = self.critics[0](whole_states_var[b], whole_newactor_actions_var[b], states_var[b, a, :],
                                               newactor_actions_var[b, a, :])
                        Qselected.append(perQ * is_weights[b])
                if len(Qselected) == 0:  # if if all tasks were allocated locally, the feedback should be sent using the commbined local decision and a fake perAgent that learns the best Q value for that situation
                    perQ = self.critics[0](whole_states_var[b], whole_newactor_actions_var[b],
                                           torch.zeros(self.state_dim), torch.zeros(self.action_dim))
                    actor_loss.append(perQ * is_weights[b])
                else:  # the best Q-value is found from the Q-vlaue of one of the selected actions
                    actor_loss.append(max(Qselected))
            actor_loss = torch.stack(actor_loss, dim=0)
            actor_loss = - actor_loss.mean()
            actor_loss.requires_grad_(True)
            self.actors_optimizer[agent_id].zero_grad()
            actor_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actors_optimizer[agent_id].step()
            self._soft_update_target(self.actors_target[agent_id], self.actors[agent_id])  # update target network
        for i in range(self.batch_size):
            idx = idxs[i]
            # print("errors",idx,errors)
            self.memory.update(idx, errors[i])
        print(
            f"[训练状态] Episode {self.n_episodes} | "
            f"Critic Loss: {critic_loss.item():.4f} | "
            f"Actor Loss: {actor_loss.mean().item():.4f} | "
            f"Mean Reward: {np.mean(self.Training_episode_rewards[-10:]):.2f}"
        )
        '''
        checkpoint = torch.load('./checkpoint/models_checkpoint'+str(self.InfdexofResult)+'.pth')    
        # Check for parameter differences in actors
        changes = []
        for agent_id in range(self.n_agents):
            ce = self.check_parameter_difference(self.actors[agent_id], checkpoint['actors'][agent_id])
            changes.append(ce)
        # Check for parameter differences in critics
        for agent_id in range(1):
            ce = self.check_parameter_difference(self.critics[agent_id], checkpoint['critics'][agent_id])
            changes.append(ce)
        if sum(changes) >1:
            #print("Model update detected", changes)
            self.save_models('./checkpoint/models_checkpoint'+str(self.InfdexofResult)+'.pth')
        elif sum(changes) == 1:
            print("No actor model update detected", changes)
            self.save_models('./checkpoint/models_checkpoint'+str(self.InfdexofResult)+'.pth')
            #exit()
        else:
            print("No model update detected", changes)
            self.save_models('./checkpoint/models_checkpoint'+str(self.InfdexofResult)+'.pth')
            #exit()
        '''

    def save_models(self, path):
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'actors_target': [actor_target.state_dict() for actor_target in self.actors_target],
            'critics': [critic.state_dict() for critic in self.critics],
            'critics_target': [critic_target.state_dict() for critic_target in self.critics_target],
        }
        torch.save(checkpoint, path)

    def check_parameter_difference(self, model, loaded_state_dict):
        current_state_dict = model.state_dict()
        for name, param in current_state_dict.items():
            if name in loaded_state_dict:
                if not torch.equal(param, loaded_state_dict[name]):
                    # print(f"Parameter '{name}' has changed since the last checkpoint.")
                    return 1
                else:
                    # print(f"Parameter '{name}' has not changed since the last checkpoint.")
                    return 0
            else:
                print(f"Parameter '{name}' is not present in the loaded checkpoint.")
                exit()

    def getactionbound(self, a, b, x, i):
        x = (x - a) * (self.action_higher_bound[i] - self.action_lower_bound[i]) / (b - a) \
            + self.action_lower_bound[i]
        return x

    # choose an action based on state with random noise added for exploration in training
    def choose_action(self, state, evaluation):
        state_var = to_tensor_var([state], self.use_cuda)

        # 初始化动作数组
        actor_action = np.zeros((self.n_agents, self.action_dim))
        critic_action = np.zeros(self.n_agents)
        hybrid_action = np.zeros((self.n_agents, self.action_dim))

        # 获取原始actor动作
        for agent_id in range(self.n_agents):
            action_var = self.actors[agent_id](state_var[:, agent_id, :])
            if self.use_cuda:
                actor_action[agent_id] = action_var.data.cpu().numpy()[0]
            else:
                actor_action[agent_id] = action_var.data.numpy()[0]

        # 探索噪声
        if not evaluation:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-1. * self.n_episodes / self.epsilon_decay)
            noise = np.random.randn(self.n_agents, self.action_dim) * epsilon
            actor_action += noise
            actor_action = np.clip(actor_action, -1, 1)

        # 初始化混合动作
        hybrid_action = deepcopy(actor_action)

        # 服务器选择逻辑
        if ENV_MODE == "H2":
            constraints = K_CHANNELS  # 每个服务器的信道约束
        elif ENV_MODE == "TOBM":
            constraints = N_UNITS  # 每个服务器的处理单元约束
        else:
            print("Unknown env_mode ", ENV_MODE)
            exit()

        # 统计每个服务器的任务申请情况
        server_requests = [[] for _ in range(N_SERVERS)]
        for agent_id in range(self.n_agents):
            # 找到最大概率的服务器选择
            server_choice = np.argmax(actor_action[agent_id, :N_SERVERS + 1])
            if server_choice > 0:  # 选择了某个服务器
                server_requests[server_choice - 1].append(agent_id)

        # 处理每个服务器的任务分配（实现论文公式21的Match_n,m）
        for s in range(N_SERVERS):
            if len(server_requests[s]) == 0:
                continue

            # 计算每个候选任务在该服务器的匹配度分数（公式21）
            # Match_n,m = α*(1/(1+|z_n-z_m^e|/2)) + β*(1/(1+|c_n-U_m^cpu*T|)) + γ*(1/(1+|η_n-η_m,opt|))
            alpha, beta, gamma = 0.4, 0.4, 0.2  # 论文中的匹配权重
            match_scores = {}
            
            # 服务器最优权衡系数
            eta_m_opt = 0.06 if s == 0 else 0.04  # 核心服务器0.06，轻量服务器0.04
            
            for agent_id in server_requests[s]:
                z_n = state[agent_id, 3]  # 任务大小
                c_n = state[agent_id, 4]  # 任务计算需求
                local_res = state[agent_id, 6]  # 本地计算资源
                
                # 存储匹配项：1/(1+|z_n-z_m^e|/2)
                storage_term = 1.0 / (1.0 + abs(z_n - S_ES[s]) / 2.0)
                
                # 计算能力匹配项：1/(1+|c_n-U_m^cpu*T|)
                # 这里简化为本地资源与服务器能力的匹配
                compute_term = 1.0 / (1.0 + abs(local_res - CAPABILITY_ES[s]))
                
                # 权衡系数匹配项：1/(1+|η_n-η_m,opt|)
                # 这里使用任务大小和截止时间的比值作为η_n的近似
                eta_n = z_n / (state[agent_id, 5] + 1e-9)  # 简化的η_n计算
                tradeoff_term = 1.0 / (1.0 + abs(eta_n - eta_m_opt))
                
                # 综合匹配分数
                match_scores[agent_id] = alpha * storage_term + beta * compute_term + gamma * tradeoff_term

            # 检查是否超过约束（信道/处理单元或存储）
            if len(server_requests[s]) > constraints[s] or \
                    np.sum(state[server_requests[s], 3]) > S_ES[s]:

                if not evaluation and (np.random.rand() <= epsilon):  # 探索
                    # 随机选择满足约束的任务（约束内化设计）
                    agent_list = deepcopy(server_requests[s])
                    random.shuffle(agent_list)
                    selected = []
                    total_size = 0
                    for agent_id in agent_list:
                        if len(selected) < constraints[s] and \
                                total_size + state[agent_id, 3] <= S_ES[s]:
                            selected.append(agent_id)
                            total_size += state[agent_id, 3]
                    # 更新critic_action
                    for agent_id in server_requests[s]:
                        critic_action[agent_id] = 1 if agent_id in selected else 0
                else:
                    # Per-action DQN决策机制：为每个任务请求独立生成Q值
                    states_var = to_tensor_var(state, self.use_cuda).view(-1, self.n_agents, self.state_dim)
                    whole_states_var = states_var.view(-1, self.n_agents * self.state_dim)
                    actor_action_var = to_tensor_var(actor_action, self.use_cuda).view(-1, self.n_agents,
                                                                                       self.action_dim)
                    whole_actions_var = actor_action_var.view(-1, self.n_agents * self.action_dim)

                    # 计算每个请求的Q值（Per-action DQN）
                    raw_qs = []
                    for agent_id in server_requests[s]:
                        qv = self.critics[0](
                            whole_states_var.squeeze(),
                            whole_actions_var.squeeze(),
                            states_var[0, agent_id, :],
                            actor_action_var[0, agent_id, :]
                        ).detach().cpu().numpy()
                        raw_qs.append(qv)
                    raw_qs = np.array(raw_qs).astype(np.float32)
                    
                    # 归一化Q值到[-10, 10]范围
                    if raw_qs.size == 0:
                        norm_qs = np.zeros(0)
                    else:
                        q_min, q_max = np.min(raw_qs), np.max(raw_qs)
                        norm_qs = (raw_qs - q_min) / (q_max - q_min + 1e-9) * 20 - 10  # 映射到[-10,10]

                    # 综合Q值与匹配度得分
                    combined = []
                    for k, agent_id in enumerate(server_requests[s]):
                        q_part = norm_qs[k] if norm_qs.size > 0 else 0.0
                        # 论文中Q值与匹配度的综合
                        combined.append((agent_id, 0.5 * q_part + 0.5 * match_scores[agent_id]))

                    # 按组合得分排序并选择
                    sorted_pairs = sorted(combined, key=lambda x: x[1], reverse=True)
                    selected = []
                    total_size = 0
                    for agent_id, _ in sorted_pairs:
                        if len(selected) < constraints[s] and \
                                total_size + state[agent_id, 3] <= S_ES[s]:
                            selected.append(agent_id)
                            total_size += state[agent_id, 3]
                            critic_action[agent_id] = 1
                        else:
                            critic_action[agent_id] = 0
            else:
                # 未超过并且仍需考虑存储：若总大小超过存储则按 Match 选取
                total_size = np.sum(state[server_requests[s], 3])
                if total_size <= S_ES[s]:
                    for agent_id in server_requests[s]:
                        critic_action[agent_id] = 1
                else:
                    sorted_by_match = sorted(server_requests[s], key=lambda aid: match_scores[aid], reverse=True)
                    selected = []
                    acc = 0
                    for agent_id in sorted_by_match:
                        if len(selected) < constraints[s] and acc + state[agent_id, 3] <= S_ES[s]:
                            selected.append(agent_id)
                            acc += state[agent_id, 3]
                            critic_action[agent_id] = 1
                        else:
                            critic_action[agent_id] = 0

        # 构建最终混合动作
        for n in range(self.n_agents):
            # 服务器选择部分
            if critic_action[n] == 1:  # 允许卸载
                # 保留原始服务器选择概率
                pass
            else:  # 本地执行
                hybrid_action[n, :N_SERVERS + 1] = 0
                hybrid_action[n, 0] = 1  # 设置为本地执行

            # 资源和功率部分 (保持原逻辑)
            b = 1
            a = -b
            for i in range(N_SERVERS + 1, self.action_dim):
                hybrid_action[n][i] = self.getactionbound(a, b, hybrid_action[n][i], i)

        return actor_action, critic_action, hybrid_action

    def evaluate(self, EVAL_EPISODES):
        print(f"[评估开始] 共需评估 {EVAL_EPISODES} 次")
        start_time = time.time()

        # 初始化约束检查变量
        server_constraints = K_CHANNELS if ENV_MODE == "H2" else N_UNITS
        server_violations = [0] * N_SERVERS  # 每个服务器的约束违反次数

        for i in range(EVAL_EPISODES):
            self.eval_env_state = self.env_eval.reset_mec(i)
            self.eval_step_rewards = []
            self.server_step_constraint_exceeds = 0
            self.energy_step_constraint_exceeds = 0
            self.time_step_constraint_exceeds = 0
            done = False

            while not done:
                state = self.eval_env_state
                actor_action, critic_action, hybrid_action = self.choose_action(state, True)

                # 检查每个服务器的约束违反情况
                for s in range(N_SERVERS):
                    # 获取选择该服务器的任务索引
                    server_tasks = np.where(hybrid_action[:, s + 1] > 0.5)[0]  # 阈值判断
                    task_count = len(server_tasks)
                    total_size = np.sum(state[server_tasks, 3]) if task_count > 0 else 0

                    # 检查是否违反约束
                    if task_count > server_constraints[s] or total_size > S_ES[s]:
                        server_violations[s] += 1

                next_state, reward, done, eneryconstraint_exceeds, timeconstraint_exceeds = \
                    self.env_eval.step_mec(hybrid_action)

                self.eval_step_rewards.append(np.mean(reward))
                self.energy_step_constraint_exceeds += eneryconstraint_exceeds
                self.time_step_constraint_exceeds += timeconstraint_exceeds

                if done:
                    # 计算平均违反率 (所有服务器的平均值)
                    avg_server_violation = np.mean([
                        v / len(self.eval_step_rewards) for v in server_violations
                    ])

                    self.eval_episode_rewards.append(np.sum(np.array(self.eval_step_rewards)))
                    self.server_episode_constraint_exceeds.append(avg_server_violation)
                    self.energy_episode_constraint_exceeds.append(
                        self.energy_step_constraint_exceeds / len(self.eval_step_rewards))
                    self.time_episode_constraint_exceeds.append(
                        self.time_step_constraint_exceeds / len(self.eval_step_rewards))

                    # 重置统计
                    self.eval_step_rewards = []
                    self.server_step_constraint_exceeds = 0
                    self.energy_step_constraint_exceeds = 0
                    self.time_step_constraint_exceeds = 0
                    server_violations = [0] * N_SERVERS

                    if self.done_penalty is not None:
                        reward = self.done_penalty
                else:
                    self.eval_env_state = next_state

            # 最后一轮评估完成后的处理 (与原逻辑相同)
            if i == EVAL_EPISODES - 1 and done:
                mean_reward = np.mean(np.array(self.eval_episode_rewards))
                mean_constraint = np.mean(np.array(self.server_episode_constraint_exceeds))
                mean_energyconstraint = np.mean(np.array(self.energy_episode_constraint_exceeds))
                mean_timeconstraint = np.mean(np.array(self.time_episode_constraint_exceeds))

                # ... 其余保存和打印逻辑保持不变 ...
                self.eval_episode_rewards = []
                self.server_episode_constraint_exceeds = []
                self.energy_episode_constraint_exceeds = []
                self.time_episode_constraint_exceeds = []
                self.mean_rewards.append(mean_reward)  # to be plotted by the main function
                self.episodes.append(self.n_episodes + 1)
                self.results.append(mean_reward)
                self.serverconstraints.append(mean_constraint)
                self.energyconstraints.append(mean_energyconstraint)
                self.timeconstraints.append(mean_timeconstraint)
                arrayresults = np.array(self.results)
                arrayserver = np.array(self.serverconstraints)
                arrayenergy = np.array(self.energyconstraints)
                arraytime = np.array(self.timeconstraints)
                savetxt('./CSV/results/ECCC-MADRL' + str(self.InfdexofResult) + '.csv', arrayresults)
                savetxt('./CSV/Server_constraints/ECCC-MADRL' + str(self.InfdexofResult) + '.csv', arrayserver)
                savetxt('./CSV/Energy_constraints/ECCC-MADRL' + str(self.InfdexofResult) + '.csv', arrayenergy)
                savetxt('./CSV/Time_constraints/ECCC-MADRL' + str(self.InfdexofResult) + '.csv', arraytime)
                print("Episode:", self.n_episodes, "Episodic Energy:  Min mean Max : ", np.min(arrayenergy),
                      mean_energyconstraint, np.max(arrayenergy))
                print(f"[评估完成] 耗时 {time.time() - start_time:.2f}秒")  # 新增

    def evaluateAtTraining(self, EVAL_EPISODES):
        # print(self.eval_episode_rewards)
        mean_reward = np.mean(np.array(self.Training_episode_rewards))
        self.Training_episode_rewards = []
        # self.mean_rewards.append(mean_reward)# to be plotted by the main function
        self.Training_episodes.append(self.n_episodes + 1)
        self.Training_results.append(mean_reward)
        arrayresults = np.array(self.Training_results)
        savetxt('./CSV/AtTraining/ECCC-MADRL' + self.InfdexofResult + '.csv', arrayresults)
        # print("Episode:", self.n_episodes, "Episodic Reward:  Min mean Max : ", np.min(arrayresults), mean_reward, np.max(arrayresults))
