from ray.rllib.env import MultiAgentEnv
from .envwrapper import wrap_env
import grid2op
from lightsim2grid import LightSimBackend
import numpy as np
from .my_reward import get_reward
from grid2op.Reward import CombinedReward
from grid2op.Parameters import Parameters

def env_creator(env_config):
    p = Parameters()
    p.NB_TIMESTEP_COOLDOWN_SUB = 10
    p.NB_TIMESTEP_COOLDOWN_LINE = 10
    env = grid2op.make('l2rpn_2019', backend=LightSimBackend(), reward_class=CombinedReward, param=p)
    env = get_reward(env)
    env.seed(np.random.randint(1,50))
    env = wrap_env(env,"/home/xpwang/data/proj_doc_3/hierarchical/config/all_action.json")
    env = MyEnv(env)
    env.reset()
    return env

class MyEnv(MultiAgentEnv):
    def __init__(self, env, gamma=0.99):
        super(MyEnv, self).__init__()
        self.base_env = env
        self.max_low_step = 5
        self.gamma = gamma

    def reset(self, **kwargs):
        ob = self.base_env.reset(**kwargs)
        self.last_high_ob = ob
        self.low_agent_num = 0
        self.low_steps = 0
        self.low_nothing =True
        self.reward_cum = 0
        return {"high_level_agent": ob}


    def step(self,action):
        if "high_level_agent" in action:
            if action["high_level_agent"] == 0:
                o_ob, o_r, o_done, o_info = self.base_env.step(action["high_level_agent"])
                self.last_high_ob = o_ob
                ob = {"high_level_agent":o_ob}
                r = {"high_level_agent": o_r}
                done = {"__all__":o_done}
            else:
                ob = {"low_level_agent_" + str(self.low_agent_num): self.last_high_ob}
                done = {"__all__":False, "low_level_agent_" + str(self.low_agent_num):False}
                r = {"low_level_agent_" + str(self.low_agent_num): 0}
        else:
            o_ob, o_r, o_done, o_info = self.base_env.step(action["low_level_agent_" + str(self.low_agent_num)])
            done = {"__all__": o_done}
            if o_r > 0:
                self.low_nothing = False
            ob = {"low_level_agent_" + str(self.low_agent_num):o_ob}
            r = {"low_level_agent_" + str(self.low_agent_num):o_r}
            self.reward_cum += self.gamma ** self.low_steps * o_r
            self.low_steps += 1
            if o_done:
                ob["high_level_agent"] = o_ob
                r["high_level_agent"] = self.reward_cum
            else:
                if self.low_steps == self.max_low_step:
                    done["low_level_agent_" + str(self.low_agent_num)] = True
                    ob["high_level_agent"] = o_ob
                    if self.low_nothing:
                        o_r = -1
                        r["low_level_agent_" + str(self.low_agent_num)] = o_r
                        r["high_level_agent"] = o_r
                    else:
                        r["high_level_agent"] = self.reward_cum
                    self.low_steps = 0
                    self.low_nothing = True
                    self.low_agent_num += 1
                    self.reward_cum = 0
        info = {}
        return ob, r, done, info


if __name__ == '__main__':
    pass