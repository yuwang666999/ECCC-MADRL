import numpy as np
from copy import deepcopy

# 根据论文更新参数
LAMBDA_E = 0.5  # λ1 - 能耗权重系数
LAMBDA_T = 0.5  # λ2 - 时间权重系数
MIN_SIZE = 1  # z_n 最小值 (MB)
MAX_SIZE = 50  # z_n 最大值 (MB)
MAX_CYCLE = 737.5  # C_n 最大值 (cycles/bit)
MIN_CYCLE = 300  # C_n 最小值 (cycles/bit)
MIN_DDL = 0.1  # T_n 最小值 (seconds)
MAX_DDL = 0.9  # T_n 最大值 (seconds) - 论文中为[0.1, 0.9]
MIN_RES = 0.4  # f_n,min (GHz)
MAX_RES = 1.5  # f_n,max 最大值 (GHz)
MAX_POWER = 24  # P_n,max 最大值 (dBm)
MIN_POWER = 1  # P_n,min 最小值 (dBm)
K_ENERGY_LOCAL = 5e-27  # k_local 本地能耗系数 (MJ)
K_ENERGY_SERVER = 8e-27  # k_server 服务器能耗系数 (MJ)
MAX_ENE = 3.2  # b_n,max 电池容量最大值 (MJ)
MIN_ENE = 0.5  # b_n,min 电池容量最小值 (MJ)
# 能量收集参数
HARVEST_MIN = 0.0005  # e_n(t) 最小值 (MJ)
HARVEST_MAX = 0.0015  # e_n(t) 最大值 (MJ)
SELF_DISCHARGE_RATE = 0.005  # 电池自放电率
MAX_GAIN = 14  # g_{n,m} 最大值 (dB)
MIN_GAIN = 5  # g_{n,m} 最小值 (dB)
W_BANDWIDTH = 40  # B_total 基站总带宽 (MHz)
ETA_MAX = 0.05  # η_{n,max} 能量-时间权衡系数阈值 (MJ/s)

# 根据论文Table 2更新多服务器参数
N_SERVERS = 3  # M = 3 (1个核心, 2个轻量)
# 核心服务器: K_m=15, 轻量服务器: K_m=10
K_CHANNELS = [15, 10, 10]  # 每个服务器的子信道数
# 核心服务器: Z_m^e=800MB, 轻量服务器: Z_m^e=400MB
S_ES = [800, 400, 400]  # 每个服务器的存储容量(MB)
# 核心服务器: U_m=16, 轻量服务器: U_m=8
N_UNITS = [16, 8, 8]  # 每个服务器的处理单元数
# 核心服务器: f_m^cpu=6GHz, 轻量服务器: f_m^cpu=4GHz
CAPABILITY_ES = [6, 4, 4]  # 每个服务器的计算能力(GHz)

ENV_MODE = "H2"  # ["H2", "TOBM"]
# MAX_STEPS = 10
MAX_STEPS = 30

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
        # 每个设备的最大电量 b_n,max（式11）
        self.B_max = np.zeros(self.n_agents)

    def reset_mec(self, eval_env_seed=None):
        if eval_env_seed is not None:
            np.random.seed(eval_env_seed)

        self.step = 0
        for n in range(self.n_agents):
            self.S_size[n] = np.random.uniform(MIN_SIZE, MAX_SIZE)
            self.S_cycle[n] = np.random.uniform(MIN_CYCLE, MAX_CYCLE)
            self.S_ddl[n] = np.random.uniform(MIN_DDL, MAX_DDL - MAX_DDL / 10)
            self.S_energy[n] = deepcopy(self.Initial_energy[n])
            self.B_max[n] = np.random.uniform(MIN_ENE, MAX_ENE)

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

        # 更新电池电量 b_n(t+1)（式11）：
        # b_{t+1} = clip_{[b_min, b_max]}( b_t*(1-0.005) - E_n(MJ) + e_n )
        E_MJ = Energy_n * 1e-6
        harvest = np.random.uniform(HARVEST_MIN, HARVEST_MAX, size=self.n_agents)
        b_next = self.S_energy * (1 - SELF_DISCHARGE_RATE) - E_MJ + harvest
        # 先按各自 b_max 上界截断，再保证不低于 b_min
        b_next = np.minimum(b_next, self.B_max)
        b_next = np.maximum(b_next, MIN_ENE)
        self.S_energy = b_next

        # 统计违反次数（用于可视化）
        ddl_violations = (np.array(Time_finish) > self.S_ddl).astype(np.float32)
        battery_violations = (self.S_energy < MIN_ENE + 1e-9).astype(np.float32)
        time_penalty_nonzero_count = float(np.sum(ddl_violations)) / self.n_agents
        energy_penalty_nonzero_count = float(np.sum(battery_violations)) / self.n_agents

        # 能量-时间权衡系数 η_n（式13）
        # η = E(MJ) / t(s)
        eta = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            eta[i] = E_MJ[i] / max(Time_finish[i], 1e-9)

        # 奖励函数（式15）：R = - (λ1 E_n(t) + λ2 t_n(t)) + r_penalty
        base_cost = LAMBDA_E * E_MJ + LAMBDA_T * np.array(Time_n)
        r_penalty = np.zeros(self.n_agents)
        
        # 约束违反惩罚（根据论文）
        # 如果 b_n(t+1) < b_n,min: r_penalty = -50
        r_penalty[self.S_energy <= MIN_ENE + 1e-9] += -50.0
        
        # 如果 t_n(t) > T_n: r_penalty = -100
        r_penalty[np.array(Time_finish) > self.S_ddl] += -100.0
        
        # 如果 η_n > η_n,max: r_penalty = -(η_n - η_n,max) × 10
        eta_excess = np.maximum(eta - ETA_MAX, 0.0)
        r_penalty += -10.0 * eta_excess
        
        # 如果无约束违反: r_penalty = 0 (已经初始化为0)

        Reward = -base_cost + r_penalty
        # 与原实现保持一致：各体奖励设为总和
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