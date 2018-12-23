import gym
from gym.spaces import Box, Discrete
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise

from vrep_env import VrepEnv

import errno
import os
from datetime import datetime

from actor_net import ActorNet
from critic_net import CriticNet

import argparse

from api import vrep

episodes= 100
 #batch normalization switch

def main():
    env = VrepEnv()
    agent = DDPG(env, is_batch_norm=False)


    exploration_noise = OUNoise(env.action_space.shape[0])
    counter=0
    reward_per_episode = 0    
    total_reward=0
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]    

    print ("Number of States:", num_states)
    print ("Number of Actions:", num_actions)

    #saving reward:
    reward_st = np.array([0])
    step = 0

    try:
        agent.load_model()
    except:
        pass
    

    for i in range(episodes):
        print ("================= Starting episode no:",i,"==================","\n")
        observation = env.reset()

        #print(observation)

        reward_per_episode = 0

        while True:

            x = observation
            #print ('x: ', x)
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            #print('??????:', action)
            noise = exploration_noise.noise()

            #print('noise:', noise)
            #print('noise2:', noise2)

            noise = noise[0]*0.2
            action = action[0]
            #print('action pre:', action)
            action = action + noise


            #print "Action at step", t ," :",action,"\n"
            observation,reward,done=env.step(action)

            #add s_t,s_t+1,action,reward to experience memory
            agent.add_experience(x,observation,action,reward, done)
            #train critic and actor network
            if counter > 64: 
                agent.train()
            reward_per_episode+=reward
            counter+=1

            if env.finishCheck():
                print ('EPISODE: ',i, 'Total Reward: ',reward_per_episode)
                #print "Printing reward to file"
                exploration_noise.reset()
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")

                if i % 1 == 0:
                    print('save')
                    agent.save_model()

                break

            step += 1

    total_reward+=reward_per_episode            
    print ("Average reward per episode {}".format(total_reward / episodes)    )
    env = VrepEnv()



if __name__ == '__main__':
    main()  

       
