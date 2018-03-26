import gym, acme_gym
import numpy as np
import time as time

def simple_test(pause=0.001, steps=100, random=True):
    env = gym.make("CartPoleContinuous-v0")
    env.seed(0)
    obs = env.reset()
    print("Initial obs", obs)
    env.seed(0)
    obs2 = env.reset()
    print("New Obs:", obs2)
    return
    keep_going = False
    env.render()
    for i in range(steps):
        if random is True:
            step = np.random.uniform(low=-10, high=10)
        else:
            print("Current state is:", obs) 
            step = input("Please enter a float, or q to quit: ")
            if step == 'q':
                return
            step = float(step)
        # Make a step in the right direction
        step = np.array([step])
        obs, rew, done, _ = env.step(step)
        env.render()
        
        time.sleep(pause)
        if done and keep_going is False:
            print("You tipped over or moved too far!")
            q = input("Type y to keep going, all else quits: ")
            if q.lower() != 'y':
                return
            else:
                keep_going = True
    input("Enter anything to quit")

if __name__ == "__main__":
    simple_test(random=False)