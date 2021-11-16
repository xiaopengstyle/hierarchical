from grid2op.Exceptions import Grid2OpException
from grid2op.Reward import LinesCapacityReward,CombinedReward

def get_reward(env):
    cr: CombinedReward = env.get_reward_instance()
    cr.addReward("LinesCapacity", LinesCapacityReward(),1)
    cr.initialize(env)
    return env