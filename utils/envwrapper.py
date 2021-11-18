from gym.core import ObservationWrapper,ActionWrapper,RewardWrapper
import gym
import numpy as np
from grid2op.Converter import AnalogStateConverter
#用env._thermal_limit_a的最大值来对a进行归一化；用最大的gen_pmax来对P,Q进行归一化；
#如果不进行发电机调控，则不需要redispatching的数据
#to_vect，先加入时间信息，然后按照下面的顺序进行组织
observation_keys = ['gens',  #float,有功，无功，电压；用最大的gen_pmax来进行归一化 3*env.n_gen
                    'loads',  #float,有功，无功，电压;3*env.n_load
                    'lines_or',  # p,q,v,a；用env._thermal_limit_a的最大值来进行归一化;4*env.n_line
                    'lines_ex',  # 4*env.n_line
                    'rho',  #float，本身范围就在0~1;env.n_line
                    'line_status',  #bool，线路状态；env.n_line
                    'timestep_overflow',  #int,线路过负荷的时间,用env.nb_timestep_overflow_allowed归一化; env.n_line
                    'topo_vect',  #int,连接的bus编号，0，1，2；env.dim_topo
                    'cooldown',  #冷却时间,env.n_line+env.n_sub
                    'maintenance',  #time_next_maintenance:-1,不会检修，0，已经检修断开了，>1，,duration_next_maintenance，还需要持续的时间;2*env.n_line
                    'redispatching',  #调控,2*env.n_gen
                    ]
observation_second_keys = {'loads':['p', 'q', 'v'],'gens':['p', 'q', 'v'],'lines_or':['p', 'q', 'v','a'],
                           'lines_ex':['p', 'q', 'v', 'a'],'cooldown':['line', 'substation'],
                           'maintenance':['time_next_maintenance', 'duration_next_maintenance'],
                           'redispatching':['target_redispatch', 'actual_dispatch']
                           }

def wrap_env(env, path):
    return MyActionWrapper(MyObservationWrapper(env), path)


class MyObservationWrapper(ObservationWrapper):
    def __init__(self,env,with_redispatch=False):
        super(MyObservationWrapper, self).__init__(env)
        # 需要找到每个值的边界，然后在此处定义观测空间，此处对需要观察的关键词进行定义
        #如果需要时间信息,通过observation.hour_of_day，observation.day_of_week,observation.minute_of_hour
        #observation.day,observation.month,observation.
        self.original_env = env
        self.start_index = 3

        self.observation_list = [
            "prod_p","prod_q","prod_v",
            "load_p","load_q","load_v",
            "p_or","q_or","v_or","a_or",
            "p_ex","q_ex","v_ex","a_ex",
            "rho",
        ]
        self.obs_dim = 3 * env.n_gen  + 3 * env.n_load + 9 * env.n_line
        self.observation_space = gym.spaces.Box(np.ones(self.obs_dim)*(-1e6),np.ones(self.obs_dim)*(1e6),dtype=np.float)



    def observation(self, observation):
        self.obs = observation
        li_vect = [observation._get_array_from_attr_name(el).astype(np.float) for el in self.observation_list]
        if li_vect:
            return np.concatenate(li_vect)
        else:
            return np.array([], dtype=np.float)
        # return {'load':load,'gen':gen,'gen_dispatch':dispatch,'sub':cooldown_sub,'topo':topo,
        #         'line_float':np.vstack([line_ex,line_or,rho]),
        #         'line_int':np.vstack([line_maintain,line_status,line_overflow,cooldown_line]),



class MyActionWrapper(ActionWrapper):
    def __init__(self, env, path):
        super(MyActionWrapper, self).__init__(env)
        self.action_env = env
        self.actions, self.vect_actions = self.get_vects(env.action_space, path)
        self.action_space = gym.spaces.Discrete(len(self.actions))
        # self.with_dispatch = with_dispatch

    def step(self,o_action):
        ob = self.original_env.current_obs
        _reward = True
        action = self.action(o_action)
        observation, reward, done, info = self.env.step(action)
        if self.original_env.nb_time_step < ob.max_step and done:
            reward = -10
        elif done:
            reward = 10
        else:
            if o_action != 0:
                if ob + action == ob + self.original_env.action_space({}):
                    reward = -0.01
            else:
                reward = 0
        return observation,reward,done,info


    def action(self, action):
        the_action = self.actions[action]
        return self.original_env.action_space(self.process_action(self.obs, the_action))

    def get_vects(self, action_space, path):
        import json
        with open(path) as f:
            actions = json.load(f)
        vect_actions = []
        vect_actions.append({0: action_space({}).to_vect()})
        for k, v in enumerate(actions):
            if isinstance(v, dict):
                vect_actions.append({k: action_space(v).to_vect()})
        return actions, vect_actions

    def process_action(self,ob,action):
        if action == "do_nothing":
            return {}
        elif action == "reset_redispatch":
            act = []
            for i in range(len(ob.target_dispatch)):
                if ob.gen_redispatchable[i]:
                    act.append([i,-ob.target_dispatch[i]])
            return {"redispatch":act}
        elif action == "reconnect_lines":
            act = []
            for i in range(ob.n_line):
                if ob.time_before_cooldown_line[i] <= 0 and not ob.line_status[i]:
                    act.append([i,+1])
                    break
            return {"set_line_status":act}
        return action

if __name__ == '__main__':
    import grid2op
    from grid2op.PlotGrid import PlotMatplot
    import matplotlib.pyplot as plt
    env = wrap_env(grid2op.make())
    fig = plt.figure()

    plthelper = PlotMatplot(env.observation_env.observation_space)
    for i in range(env.action_space.n):
        env.reset()
        obs,_,_,_ = env.step(i)
        plthelper.plot_obs(env.observation_env.current_obs,fig)
        # cv2.waitKey(10)
        plt.show()


