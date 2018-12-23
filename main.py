
from vrep_env import VrepEnv
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise

def main():   

    env = VrepEnv()
    num_states = env.observation_space.shape[0]
    agent = DDPG(env)
    agent.load_model()
    observation = env.reset()
    exploration_noise = OUNoise(env.action_space.shape[0])

    while True:
        x = observation
        action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
        noise = exploration_noise.noise()
        action = action[0] + noise*0.2
        observation,reward, done = env.step(action)
        agent.add_experience(x,observation,action,reward,done)
if __name__ == '__main__':
    main()