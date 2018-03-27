This repository was prepared as part of the Midterm and Final Project for BYU's Math 438 class, Modeling with Dynamics and Control 2.
Students were put into groups of 4 and asked to solve the continuous cartpole problem by using some form of Optimal Control.  They will have succeeded if they can demonstrate robust code that manages to keep/bring the pendulum upright and remain there "for all time".

# ACME-GymEnvs

### Barebones Installation Guide
The Environments here were initially built based off of Gym version 0.10.4, which was the latest version as of March 2018.

Make sure you have gym installed following the instructions found [here](https://github.com/openai/gym) or [here](https://gym.openai.com/).

Clone this repository and do a local installation with the following commands:
```
git clone https://MitchProbst/ACME-GymEnvs
cd ACME-GymEnvs
pip install -e
```
Note: You do not have to pip install the environment, but if you do not pip install you will need to put the [acme_gym](https://github.com/MitchProbst/ACME-GymEnvs/tree/master/acme_gym) folder in the parent directory of whatever solution file you are going to use.

You can test that you have installed it correctly by running a python script like the following:
```
import gym, acme_gym
import numpy as np

# CartPoleContinuous-v0 is one of the custom environments for you to use
env = gym.make("CartPoleContinuous-v0")
observation = env.reset()
# observation is a 4-tuple of x, x', θ, θ' 
print(observation)
env.render()

# Push the cart to the right with a force of 1
new_obs, reward, done, info = env.step(np.array([1]))
env.render()

# call env.close() at the end of a file to terminate the viewing window
```

Because Gym is meant for reinforcement learning it makes some assumptions about when you want to terminate a run, this is managed with the done variable that is returned at each period.  For the purposes of Optimal Control, we can simply ignore that, even if it brings up a warning. Reward and info are also useless variables to us.

There are two examples to illustrate how to work with the gym environment in a .py file or a .ipynb.
Working in a jupyter notebook can cause your kernel to crash when attempting to render your system, if you attempt to interrupt a notebook while the system is running. Proceed with caution, you have been warned!

See the .py example [here](https://github.com/MitchProbst/ACME-GymEnvs/blob/master/example.py).

See the jupyter example [here](https://github.com/MitchProbst/ACME-GymEnvs/blob/master/example.ipynb).

### Important Details
1) Gym is designed to step at .02 second intervals, so the time elapsed from one step to another is .02 seconds. You might have a hard time getting a solver to work right if you do not mimic the .02s stepsize (maybe you can, if so, you are awesome!)
2) I highly recommend reading the source code for the [CartPoleContinuousEnv](https://github.com/MitchProbst/ACME-GymEnvs/blob/master/acme_gym/envs/cartpole_continuous.py#L15) class, it contains relevant information like the variables of our system.
3) If you look at the source code and see length as 0.5, read the comment, it really means the total length of the pole is 1.
4) One possible solution path is to re-use some of the principles from the [ACME LQR lab](http://www.acme.byu.edu/wp-content/uploads/2018/03/21-Inverted-Pendulum.pdf), but recognize in that lab we used a "Rickshaw" that assumes a massless rod with a weighted object at the end. Here we are using a rod with mass that has no additional object at the end.
5) You might also want to figure out how to implement some kind of continuous feedback into your model, you may not be able to perfectly describe the dynamics of the system in any solver you use, but you hopefully can get close.
6) To see all possible environments and the label called to instantiate them, look [here](https://github.com/MitchProbst/ACME-GymEnvs/blob/master/acme_gym/__init__.py).

### Some Troubleshooting
Q: I'm getting a `raise NotImplementedError('abstract') NotImplementedError: abstract error`, what should I do?
A: Try following [this](https://github.com/openai/gym/issues/775) link that suggests downgrading pyglet to version 1.2.4 and see if that fixes your problem.

Q: I am getting an `AssertionError: array(\[ BIG NUMBER ]) (<class 'numpy.ndarray'>) invalid`
A: The environment is capped so you can't apply forces of insane magnitude (2000 is already really large), if you're trying to apply forces stronger than that, you ought to be refactoring your solution.

Q: I can not see the environment when I call `env.render()` even though I am getting no errors, what is going on?
A: The viewer showing the actual CartPole may not appear at the front of your screen. Check and see if it appeared in the background somewhere, especially if you are using a notebook.
