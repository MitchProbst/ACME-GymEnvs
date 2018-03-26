import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CartPoleContinuous-v0',
    entry_point='acme_gym.envs:CartPoleContinuousEnv',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic = True,
)

register(
    id='StochasticCartPoleContinuous-v0',
    entry_point='acme_gym.envs:StochasticCartPoleContinuousEnv',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic = True,
)

# Examples of other environments
"""register(
    id='SoccerEmptyGoal-v0',
    entry_point='gym_soccer.envs:SoccerEmptyGoalEnv',
    timestep_limit=1000,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='SoccerAgainstKeeper-v0',
    entry_point='gym.envs:SoccerAgainstKeeperEnv',
    timestep_limit=1000,
    reward_threshold=8.0,
    nondeterministic = True,
)"""