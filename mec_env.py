import numpy as np
from copy import deepcopy

# 基础参数保持不变
LAMBDA_E = 0.5
LAMBDA_T = 0.5
MIN_SIZE = 1  # MB*1024*8
MAX_SIZE = 50  # MB*1024*8 #bits
MAX_CYCLE = 737.5  # cycles as in reference
MIN_CYCLE = 300  # cycles (customized)
MIN_DDL = 0.1  # seconds
MAX_DDL = 1  # seconds
MIN_RES = 0.4  # GHz*10**9 #cycles per second
MAX_RES = 1.5  # GHz*10**9 #cycles per second
MAX_POWER = 24  # 10**(24/10) # 24 dBm converting 24 dB to watt(j/s)
MIN_POWER = 1  # 10**(1/10) # converting 1 dBm to watt(j/s)
K_ENERGY_LOCAL = 5 * 1e-27  # no conversion
MAX_ENE = 3.2  # MJ*10**6 # in joules
MIN_ENE = 0.5  # MJ*10**6 # in joules
HARVEST_RATE = 0.001  # in joules
MAX_GAIN = 14  # dB
MIN_GAIN = 5  # dB
W_BANDWIDTH = 40  # MHZ

# 多服务器参数
N_SERVERS = 3  # 服务器数量
K_CHANNELS = [10, 8, 12]  # 每个服务器的信道数
S_ES = [400, 300, 500]  # 每个服务器的存储容量(MB*1024*8)
N_UNITS = [8, 6, 10]  # 每个服务器的处理单元数
CAPABILITY_ES = [4, 3.5, 5]  # 每个服务器的计算能力(GHz*10^9)

ENV_MODE = "H2"  # ["H2", "TOBM"]
MAX_STEPS = 10


class MecEnv(object):
    def __init__(self, n_agents, env_seed=None):
        if env_seed is not None:
            np.random.seed(env_seed)

        self.state_size = 7 + N_SERVERS  # 增加服务器相关信息
        self.action_size = N_SERVERS + 2
        self.n_agents = n_agents
        self.W_BANDWIDTH = W_BANDWIDTH

        # 初始化设备状态
        self.S_power = np.zeros(self.n_agents)
        self.Initial_energy = np.zeros(self.n_agents)
        self.S_energy = np.zeros(self.n_agents)
        self.S_gain = np.zeros(self.n_agents)
        self.S_size = np.zeros(self.n_agents)
        self.S_cycle = np.zeros(self.n_agents)
        self.S_ddl = np.zeros(self.n_agents)
        self.S_res = np.zeros(self.n_agents)

        # 动作边界调整
        self.action_lower_bound = [0] * N_SERVERS + [0.01, 0.01]
        self.action_higher_bound = [1] * N_SERVERS + [1, 1]

        # 初始化设备参数
        for n in range(self.n_agents):
            self.S_power[n] = np.random.uniform(MIN_POWER, MAX_POWER)
            self.Initial_energy[n] = np.random.uniform(MIN_ENE, MAX_ENE)
            self.S_gain[n] = np.random.uniform(MIN_GAIN, MAX_GAIN)
            self.S_res[n] = np.random.uniform(MIN_RES, MAX_RES)

        # 服务器状态 (每个服务器的当前负载)
        self.server_loads = np.zeros(N_SERVERS)
        self.server_storage = [s * 1024 * 8 for s in S_ES]  # 转换为bits

    def reset_mec(self, eval_env_seed=None):
        if eval_env_seed is not None:
            np.random.seed(eval_env_seed)

        self.step = 0
        for n in range(self.n_agents):
            self.S_size[n] = np.random.uniform(MIN_SIZE, MAX_SIZE)
            self.S_cycle[n] = np.random.uniform(MIN_CYCLE, MAX_CYCLE)
            self.S_ddl[n] = np.random.uniform(MIN_DDL, MAX_DDL - MAX_DDL / 10)
            self.S_energy[n] = deepcopy(self.Initial_energy[n])

        self.S_energy = np.clip(self.S_energy, MIN_ENE, MAX_ENE)
        self.server_loads = np.zeros(N_SERVERS)  # 重置服务器负载

        # 构建状态向量 (增加服务器负载信息)
        State_ = []
        for n in range(self.n_agents):
            state_n = [
                self.S_power[n],
                self.S_gain[n],
                self.S_energy[n],
                self.S_size[n],
                self.S_cycle[n],
                self.S_ddl[n],
                self.S_res[n]
            ]
            # 添加服务器负载信息
            state_n.extend(self.server_loads)
            State_.append(state_n)

        return np.array(State_)

    def step_mec(self, action):
        # 解析动作
        A_decision = action[:, :N_SERVERS + 1]  # 包括本地执行的选择
        A_res = self.S_res * 10 ** 9 * action[:, -2]  # 倒数第二维是资源
        A_power = 10 ** ((self.S_power - 30) / 10) * action[:, -1]  # 最后一维是功率

        for n in range(self.n_agents):
            # 第一个维度是决策类型 (0=本地, 1=服务器1, 2=服务器2, ...)
            decision_idx = np.argmax(action[n][:N_SERVERS + 1])
            if decision_idx == 0:  # 本地执行
                A_decision[n, 0] = 1
            else:  # 选择某个服务器
                A_decision[n, decision_idx] = 1

            A_res[n] = self.S_res[n] * 10 ** 9 * action[n][N_SERVERS + 1]  # 资源
            A_power[n] = 10 ** ((self.S_power[n] - 30) / 10) * action[n][-1]  # 功率

        # 计算数据传输速率 (考虑多服务器)
        DataRates = []
        for s in range(N_SERVERS):
            # 每个服务器有自己的信道数
            data_rate = self.W_BANDWIDTH * 10 ** 6 * np.log(1 + A_power * 10 ** (self.S_gain / 10)) / np.log(2)
            data_rate = data_rate / K_CHANNELS[s]  # 带宽按信道数分配
            DataRates.append(data_rate)

        # 计算处理时间 (每个服务器能力不同)
        Time_procs = []
        for s in range(N_SERVERS):
            time_proc = self.S_size * 8 * 1024 * self.S_cycle / (CAPABILITY_ES[s] * 10 ** 9)
            Time_procs.append(time_proc)

        Time_local = self.S_size * 8 * 1024 * self.S_cycle / A_res
        Time_max_local = self.S_size * 8 * 1024 * self.S_cycle / (MIN_RES * 10 ** 9)

        # 计算卸载时间 (根据选择的服务器)
        Time_off = np.zeros(self.n_agents)
        for n in range(self.n_agents):
            if np.any(A_decision[n, 1:]):  # 如果选择了某个服务器
                s = np.argmax(A_decision[n, 1:])  # 获取服务器索引
                Time_off[n] = self.S_size[n] * 8 * 1024 / DataRates[s][n]

        # 处理特殊决策 (如惩罚)
        for i in range(A_decision.shape[0]):
            if np.all(A_decision[i] == 0):  # 无有效决策
                Time_off[i] = MAX_DDL
                A_decision[i, 0] = 1  # 默认本地执行

        # 计算完成时间 (考虑多服务器调度)
        Time_finish = np.zeros(self.n_agents)

        if ENV_MODE == "H2":
            # 对每个服务器单独处理
            for s in range(N_SERVERS):
                # 获取选择该服务器的任务索引
                server_tasks = np.where(A_decision[:, s + 1] == 1)[0]
                if len(server_tasks) == 0:
                    continue

                # 按卸载时间排序
                SortedOff = np.argsort(Time_off[server_tasks])
                MECtime = np.zeros(N_UNITS[s])  # 该服务器的处理单元

                for i in range(len(server_tasks)):
                    task_idx = server_tasks[SortedOff[i]]
                    if i < N_UNITS[s]:  # 前N_UNITS个任务可以立即开始
                        Time_finish[task_idx] = Time_off[task_idx] + Time_procs[s][task_idx]
                        MECtime[np.argmin(MECtime)] = Time_finish[task_idx]
                    else:  # 后续任务需要等待
                        Time_finish[task_idx] = max(Time_off[task_idx], np.min(MECtime)) + Time_procs[s][task_idx]
                        MECtime[np.argmin(MECtime)] = Time_finish[task_idx]

        elif ENV_MODE == "TOBM":
            # 并发处理模式
            for n in range(self.n_agents):
                if A_decision[n, 0] == 1:  # 本地执行
                    Time_finish[n] = Time_local[n]
                else:  # 服务器执行
                    s = np.argmax(A_decision[n, 1:])  # 获取服务器索引
                    Time_finish[n] = Time_off[n] + Time_procs[s][n]
        else:
            print(ENV_MODE, " is unknown")
            exit()

        # 计算最终时间 (归一化)
        Time_n = [min(t, MAX_DDL) / MAX_DDL for t in Time_finish]
        T_mean = np.mean(Time_n)

        # 计算能耗
        Energy_local = K_ENERGY_LOCAL * self.S_size * 8 * 1024 * self.S_cycle * A_res
        Energy_max_local = K_ENERGY_LOCAL * self.S_size * 8 * 1024 * self.S_cycle * (self.S_res * 10 ** 9)
        Energy_off = A_power * Time_off

        # 根据决策计算总能耗
        Energy_n = np.zeros(self.n_agents)
        for n in range(self.n_agents):
            if A_decision[n, 0] == 1:  # 本地执行
                Energy_n[n] = Energy_local[n]
            else:  # 服务器执行
                Energy_n[n] = Energy_off[n]

        # 更新能量状态 (考虑能量收集)
        self.S_energy = np.clip(
            self.S_energy - Energy_n * 1e-6 + np.random.normal(HARVEST_RATE, 0, size=self.n_agents) * 1e-6,
            0, MAX_ENE
        )

        # 能量耗尽惩罚
        for i in range(self.n_agents):
            if self.S_energy[i] <= 0:
                Time_n[i] = MAX_DDL / MAX_DDL

        # 计算惩罚项
        Time_penalty = np.maximum((Time_n - self.S_ddl / MAX_DDL), 0)
        Energy_penalty = np.maximum((MIN_ENE - self.S_energy), 0) * 10 ** 6

        time_penalty_nonzero_count = np.count_nonzero(Time_penalty) / self.n_agents
        energy_penalty_nonzero_count = np.count_nonzero(Energy_penalty) / self.n_agents

        # 计算奖励
        Reward = -1 * (LAMBDA_E * np.array(Energy_n) + LAMBDA_T * np.array(Time_n)) \
                 - 1 * (LAMBDA_E * np.array(Energy_penalty) + LAMBDA_T * np.array(Time_penalty))
        Reward = np.ones_like(Reward) * np.sum(Reward)

        # 更新服务器负载 (用于下一状态)
        for s in range(N_SERVERS):
            server_tasks = np.where(A_decision[:, s + 1] == 1)[0]
            if len(server_tasks) > 0:
                self.server_loads[s] = np.mean([Time_finish[t] for t in server_tasks])
            else:
                self.server_loads[s] = 0

        # 生成新任务
        for n in range(self.n_agents):
            self.S_size[n] = np.random.uniform(MIN_SIZE, MAX_SIZE)
            self.S_cycle[n] = np.random.uniform(MIN_CYCLE, MAX_CYCLE)
            self.S_ddl[n] = np.random.uniform(MIN_DDL, MAX_DDL - MAX_DDL / 10)

        # 构建新状态
        State_ = []
        for n in range(self.n_agents):
            state_n = [
                self.S_power[n],
                self.S_gain[n],
                self.S_energy[n],
                self.S_size[n],
                self.S_cycle[n],
                self.S_ddl[n],
                self.S_res[n]
            ]
            # 添加服务器负载信息
            state_n.extend(self.server_loads)
            State_.append(state_n)

        self.step += 1
        done = False
        if self.step >= MAX_STEPS:
            self.step = 0
            done = True

        return np.array(State_), Reward, done, energy_penalty_nonzero_count, time_penalty_nonzero_count