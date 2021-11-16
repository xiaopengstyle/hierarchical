from ray.rllib.env import MultiAgentEnv
from .envwrapper import wrap_env
import grid2op
from lightsim2grid import LightSimBackend
import numpy as np
from .my_reward import get_reward
from grid2op.Reward import CombinedReward
from ray.rllib.agents.ppo import PPOTrainer


def env_creator(env_config):
    env = grid2op.make('l2rpn_2019', backend=LightSimBackend(), reward_class=CombinedReward, difficulty="competition")
    env = get_reward(env)
    env.seed(np.random.randint(1,50))
    env = wrap_env(env,"/home/xpwang/data/proj_doc_3/hierarchical/config/all_action.json")
    env = MyEnv(env)
    env.reset()
    return env

class MyEnv(MultiAgentEnv):
    def __init__(self, env):
        super(MyEnv, self).__init__()
        self.base_env = env

    def reset(self, **kwargs):
        ob = self.base_env.reset(**kwargs)
        self.last_high_ob = ob
        self.last_is_low = False
        self.last_low_reward = 0
        self.low_agent_num = 0
        return {"high_level_agent": ob}


    def step(self,action):
        if "high_level_agent" in action:
            if action["high_level_agent"] == 0:
                o_ob, o_r, o_done, o_info = self.base_env.step(action["high_level_agent"])
                self.last_high_ob = o_ob
                ob = {"high_level_agent":o_ob}
                r = {"high_level_agent": o_r}
                done = {"__all__":o_done}
                if self.last_is_low:
                    ob["low_level_agent_" + str(self.low_agent_num)] = o_ob
                    done["low_level_agent_" + str(self.low_agent_num)] = True
                    r["low_level_agent_" + str(self.low_agent_num)] = self.last_low_reward
                    self.low_agent_num += 1
                self.last_is_low = False
            else:
                ob = {"low_level_agent_" + str(self.low_agent_num): self.last_high_ob}
                done = {"__all__":False, "low_level_agent_" + str(self.low_agent_num):False}
                o_info = {}
                if not self.last_is_low:
                    r = {"low_level_agent_" + str(self.low_agent_num): 0}
                else:
                    r = {"low_level_agent_" + str(self.low_agent_num): self.last_low_reward}
        else:
            o_ob, o_r, o_done, o_info = self.base_env.step(action["low_level_agent_" + str(self.low_agent_num)] + 1)
            done = {"__all__": o_done}
            ob = {"high_level_agent":o_ob}
            self.last_is_low = True
            self.last_low_reward = o_r
            r = {"high_level_agent":o_r}
            if o_done:
                r["low_level_agent_" + str(self.low_agent_num)] = o_r
                ob["low_level_agent_" + str(self.low_agent_num)] = o_ob

        return ob, r, done, {}


if __name__ == '__main__':
    pass