import grid2op
from grid2op.Agent import DoNothingAgent
from lightsim2grid import LightSimBackend

if __name__ == '__main__':
    env = grid2op.make('l2rpn_2019', backend=LightSimBackend(), difficulty="competition")
    episodes = 1
    total_step = 0
    import numpy as np
    agent = DoNothingAgent(env.action_space)
    last_ob = None
    res = []
    for i in range(episodes):
        done = False
        reward = 0
        step = 0
        total_reward = 0
        env.set_id(1)
        # env.fast_forward_chronics(1000)
        obs = env.reset()
        env.fast_forward_chronics(300)
        last_ob = obs
        while not done:
            action = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(action)
            step += 1
            total_reward += reward
        total_step += step
        res.append([i,obs.max_step, step, total_reward,env.nb_time_step])
        print(res[-1])
    # np.save("episode_count",res)
    print(total_step / episodes)
