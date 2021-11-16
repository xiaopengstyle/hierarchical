import warnings
warnings.filterwarnings("ignore")
import grid2op
import ray
from lightsim2grid import LightSimBackend
from utils.myreward import CombinedReward,get_reward
from utils.envwrapper import wrap_env
from ray.tune.registry import register_env
import json
from ray import tune
from ray.tune.stopper import ExperimentPlateauStopper,MaximumIterationStopper
from ray.rllib.agents.ppo import PPOTrainer,DEFAULT_CONFIG
import numpy as np
import gym
from ray.tune import function
from ray.rllib.env import MultiAgentEnv

class MyEnv(MultiAgentEnv):
    def __init__(self, env):
        self.base_env = env
    def reset(self):
        return self.base_env.reset()
    def step(self, action):
        return self.base_env.step(action)


def env_creater(env_config):
    env = grid2op.make(backend=LightSimBackend(),reward_class =CombinedReward,**env_config)
    env = get_reward(env)
    env.seed(np.random.randint(1, 10))

    env = wrap_env(env, "/home/user/xiaopeng/proj2021/hierarchical/saved_files/all_action.json")
    env.reset()
    return MyEnv(env)

register_env("my_env",env_creater)

def load_config(path):
    config = DEFAULT_CONFIG.copy()
    with open(path) as f:
        temp = json.load(f)
    # config.update(temp)
    config.update({"env_config":temp["env_config"], "env":temp["env"], "framework": "torch"})
    return config


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    config = load_config("./config/ppo.json")
    ray.init(num_gpus=1,num_cpus=22)
    env = env_creater(config["env_config"])
    # ray.init(local_mode=True)
    def policy_mapping_fn(agent_id, **kwargs):
        if agent_id.startswith("low_level_"):
            return "low_level_agent"
        else:
            return "high_level_agent"
    config.update({
        "multiagent": {
            "policies": {
                "high_level_agent": (None, env.base_env.observation_space,
                                      gym.spaces.Discrete(3), {
                                          "gamma": 0.9
                                      }),
                "low_level_agent": (None,
                                     env.base_env.observation_space,
                                     gym.spaces.Discrete(env.base_env.action_space.n - 3), {
                                         "gamma": 0.9
                                     }),
            },
            "policy_mapping_fn": function(policy_mapping_fn),
        },
    })
    # config.update({"num_workers":1,"train_batch_size":10,"sgd_minibatch_size":5})
    env = None
    # config.update({"evaluation_interval":stop_num,"evaluation_num_episodes":30})
    analysis = tune.run("PPO",stop={"timesteps_total":int(1e6)},config=config,local_dir="./log",
                        verbose=3,reuse_actors=True,mode="max",)

    # checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial("episode_reward_mean"),metric = "episode_reward_mean")
    best_checkpoint = analysis.get_best_checkpoint(trial=analysis.get_best_trial("episode_reward_mean"),metric="episode_reward_mean",mode="max")
    print(best_checkpoint)
