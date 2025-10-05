from ECCC_MADRL import ECCC_MADDPG
import matplotlib.pyplot as plt
from mec_env import MecEnv
import sys

# 根据论文Table 2更新参数
MAX_EPISODES = 1000  # 训练轮次
EPISODES_BEFORE_TRAIN = 64  # 训练前探索轮次
NUMBER_OF_EVAL_EPISODES = 10  # 评估轮次

DONE_PENALTY = None

ENV_SEED = 37
NUMBERofAGENTS = 50  # N = 50 用户设备数量
def create_ddpg(InfdexofResult, env, env_eval, EPISODES_BEFORE_TRAIN, MAX_EPISODES):
    ecccmaddpg = ECCC_MADDPG(InfdexofResult=InfdexofResult, env=env, env_eval=env_eval, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size, 
                  action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound, episodes_before_train = EPISODES_BEFORE_TRAIN, epsilon_decay= MAX_EPISODES) 
                  
    ecccmaddpg.interact(MAX_EPISODES, EPISODES_BEFORE_TRAIN, NUMBER_OF_EVAL_EPISODES)
    return ecccmaddpg
    
def plot_ddpg(ddpg, parameter, variable="reward"):
    print("[实时日志] 开始生成绘图...")  # 新增
    plt.figure()
    if (variable == "reward"):
        for i in range(len(ddpg)):
            plt.plot(ddpg[i].episodes, ddpg[i].mean_rewards) 
            plt.xlabel("Episode")
            plt.ylabel("Reward")
                   
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(["ECCC-MADDPG"])
    save_path = f"./Figures/ECCC-MADDPG_run{parameter}.png"
    plt.savefig(save_path)
    print(f"[实时日志] 绘图已保存至 {save_path}")  # 新增
    plt.close()  # 确保释放内存

def run(InfdexofResult):
    print(f"[实时日志] 启动run()，参数={InfdexofResult}")
    env = MecEnv(n_agents=NUMBERofAGENTS, env_seed=ENV_SEED)
    print("[实时日志] 主环境初始化完成")
    eval_env = MecEnv(n_agents=NUMBERofAGENTS, env_seed=ENV_SEED)  # ENV_SEED will be reset at set()
    print("[实时日志] 评估环境初始化完成")

    # 打印相关维度信息
    print(f"[环境信息] 智能体数量 (n_agents): {env.n_agents}")
    print(f"[环境信息] 状态维度 (state_dim): {env.state_size}")
    print(f"[环境信息] 动作维度 (action_dim): {env.action_size}")
    print(f"[环境信息] 动作下界 (action_lower_bound): {env.action_lower_bound}")
    print(f"[环境信息] 动作上界 (action_higher_bound): {env.action_higher_bound}")

    ddpg = [create_ddpg(InfdexofResult, env, eval_env, EPISODES_BEFORE_TRAIN, MAX_EPISODES)]
    plot_ddpg(ddpg, "_%s" % InfdexofResult)

if __name__ == "__main__":
    InfdexofResult = sys.argv[1] # set run runnumber for indexing results, 
    run(InfdexofResult) 
