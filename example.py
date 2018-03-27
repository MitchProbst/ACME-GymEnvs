"""
Example file meant to illustrate a basic use case for one of the Environments in ACME Gym.
"""

import numpy as np

import gym
import acme_gym

def determine_step_size(mode, i, threshold=20):
    """
    A helper function that determines the next action to take based on the designated mode.
    
    Parameters
    ----------
    mode (int)
        Determines which option to choose.
    i (int)
        the current step number.
    threshold (float)
        The upper end of our control.
    
    Returns
    -------
    decision (float)
        The value to push/pull the cart by, positive values push to the right.
    """
    if mode == 1:
        return 0
    if mode == 2:
        return np.random.uniform(low=-threshold, high=threshold)
    if mode == 3:
        side = -1 if i%2 == 0 else 1
        return threshold*side
    if mode == 4:
        inp_str = "Enter a float value from -{} to {}:\n".format(threshold, threshold)
        return float(input(inp_str))

if __name__ == "__main__":
    input_str = """Enter one of the following commands:
    1) Do no action, ever
    2) Choose the direction randomly
    3) Alternate between left and right
    4) Pick the direction at each state
    5) Terminate
    """
    T = round(6/0.02)
    mode = int(input(input_str))
    if mode == 5:
        quit()
    
    env = gym.make('CartPoleContinuous-v0')
    # Initial state is drawn randomly, let the user pick a good starting point
    init_state = True
    while init_state:
        obs = env.reset()
        env.render()
        print("X: {}, X': {}, θ: {}, θ': {}".format(obs[0], obs[1], obs[2], obs[3]))
        init_state = input("Enter to accept a new value, anything else to accept:\n")
    
    for i in range(T):
        # Determine the step size based on our mode
        step = np.array([determine_step_size(mode,i)])
        
        # Step in designated direction and update the visual
        obs, reward, state, info = env.step(step)
        env.render()
    
    input("Enter any key to exit:")
    env.close()