import torch
from PPO import PPO
import pybullet_envs
import gym

save_path='models/'
env_name = "CartPole-v1"
max_training_timesteps = int(2e5)
print_steps=1600

update_steps = 1600
update_epochs = 80
save_steps=30000
eps_clip = 0.2
gamma = 0.99

actor_lr = 0.0003
critic_lr = 0.001



env=gym.make(env_name)
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n

agent=PPO(state_dim,action_dim,gamma,actor_lr,critic_lr,eps_clip,update_epochs)
step_reward=0
steps=0
print_epochs=0
print_reward=0
for step in range(1,max_training_timesteps):
    state=env.reset()
    step_reward=0
    for t in range(1,update_steps):
        action=agent.select_action(state)
        state,reward,done,_=env.step(action)
        agent.buffer.terminal.append(done)
        agent.buffer.rewards.append(reward)
        step_reward+=reward
        steps+=1
        if steps%update_steps==0:
            agent.update()
        if steps%save_steps==0:
            agent.save(save_path)
        if steps%print_steps==0:
            avg_reward=float(print_reward)/print_epochs
            print('steps:  %d, avg reward:  %.2f'%(step,avg_reward))
            print_reward=0
            print_epochs=0
        if done:
            break
    print_reward+=step_reward
    print_epochs+=1
env.close()