import torch
import time
import gym
import pybullet_envs
from PPO import PPO

save_path='models/'
env_name="CartPole-v1"
max_testing_epoch=30
render=True

env=gym.make(env_name)
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n

agent=PPO(state_dim,action_dim)
agent.load(save_path)
for i in range(max_testing_epoch):
    state=env.reset()
    reward_sum=0
    while True:
        action=agent.select_action(state)
        state,reward,done,_=env.step(action)
        reward_sum+=reward
        if render:
            env.render()
            time.sleep(0)
        if done:
            break
    agent.buffer.clear()
    print("epoch:  %d, reward:  %.2f"%(i,reward_sum))
env.close()