from ray.rllib.agents.ppo import DEFAULT_CONFIG
from utils.hier_wrapper_fix import env_creator
from ray.tune.registry import register_env
import ray
import json
import gym
from ray import tune
register_env("my_env", env_creator)

def load_config(path):
    config = DEFAULT_CONFIG.copy()
    with open(path) as f:
        temp = json.load(f)
    config.update(temp)
    # config.update({"env_config":temp["env_config"], "env":temp["env"], "framework": "torch"})
    return config



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    config = load_config("./config/ppo.json")
    ray.init(num_gpus=1,num_cpus=24)
    # ray.init(local_mode=True)
    # config.update({"num_workers":0,"num_envs_per_worker":1,})
    env = env_creator(config["env_config"])
    # ray.init(local_mode=True)
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id.startswith("low_level_"):
            return "low_level_agent"
        else:
            return "high_level_agent"
    config.update({
        "multiagent": {
            "policies": {
                "high_level_agent": (None, env.base_env.observation_space,
                                      gym.spaces.Discrete(2), {
                                        "gamma": 0.99,
                                        "entropy":0.02,
                                      }),
                "low_level_agent": (None,
                                     env.base_env.observation_space,
                                     gym.spaces.Discrete(env.base_env.action_space.n - 1), {
                                        "gamma": 0.95,
                                        "entropy": 0.01,
                                     }),
            },
            "policy_mapping_fn": policy_mapping_fn,
        },
    })
    # config.update({"num_workers":1,"train_batch_size":10,"sgd_minibatch_size":5})
    env = None
    # config.update({"evaluation_interval":stop_num,"evaluation_num_episodes":30})
    # mode = "max", metric = "episode_reward_mean"
    analysis = tune.run("PPO",stop={"timesteps_total":int(1e8)},config=config,local_dir="./log",
                        verbose=3,reuse_actors=True,keep_checkpoints_num=10,checkpoint_freq=10)