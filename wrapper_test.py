from utils.envwrapper import wrap_env
from utils.hier_wrapper_fix import MyEnv
import grid2op
from lightsim2grid import LightSimBackend
import numpy as np
from utils.my_reward import get_reward
from grid2op.Reward import CombinedReward
if __name__ == '__main__':
    env = grid2op.make('l2rpn_2019', backend=LightSimBackend(), reward_class=CombinedReward, difficulty="competition")
    env = get_reward(env)
    env.seed(np.random.randint(1,50))
    env = wrap_env(env,"config/all_action.json")
    env = MyEnv(env)
    ob = env.reset()
    episodes = 1
    total_step = 0
    for i in range(episodes):
        done = {"__all__":False}
        reward = 0
        step = 0
        total_reward = 0
        obs = env.reset()
        while not done["__all__"]:
            # if step == 10000:
            #     action = {"high_level_agent":1}
            # else:
            #     action = {"high_level_agent":0}
            has_high = False
            if "high_level_agent" in obs:
                has_high = True
            low_key = None
            for i in obs.keys():
                if "low_level_" in i:
                    low_key = i
            if low_key and not has_high:
                act = None
                while act is None:
                    try:
                        act = int(input("low action:"))
                    except:
                        act = int(input("please enter a num:"))
                action = {low_key:act}
            else:
                act = None
                while act is None:
                    try:
                        act = int(input("high action:"))
                    except:
                        act = int(input("please enter a num:"))
                action = {"high_level_agent":act}
            obs, reward, done, info = env.step(action)
            step += 1
            if "high_level_agent" in reward:
                total_reward += reward["high_level_agent"]
            print(reward)
        print(env.base_env.original_env.nb_time_step)
        total_step += step
        print(step, total_reward)
    print(total_step / episodes)