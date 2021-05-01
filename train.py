import torch
from PPO import PPO
import pybullet_envs
import gym

save_path='models/'
env_name = "CartPole-v1"
max_training_timesteps = int(2e5)
print_steps=1000
update_steps = 200
update_epochs = 80
save_steps=2000
eps_clip = 0.2
gamma = 0.99

actor_lr = 0.0003
critic_lr = 0.001



env=gym.make(env_name)
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n

agent=PPO(gamma,actor_lr,critic_lr,eps_clip,update_epochs,state_dim,action_dim)
step_reward=0
steps=0
for step in range(1,max_training_timesteps):
    state=env.reset()
    for t in range(1,update_steps):
        action=agent.select_action(state)
        state,reward,done,_=env.step(action)
        agent.buffer.terminal.append(done)
        agent.buffer.rewards.append(reward)
        step_reward+=reward
        steps+=1
        if done:
            break
    if step%update_steps==0:
        agent.update()
    if step%save_steps==0:
        agent.save(save_path)
    if step%print_steps==0:
        avg_reward=step_reward/steps
        print('steps:  %d, avg reward:  %.2f'%(step,avg_reward))
        step_reward=0
        steps=0