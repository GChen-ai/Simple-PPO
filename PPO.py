import torch
import torch.nn as nn
import torch.distributions
import numpy as np
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0')
class Buffer:
    def __init__(self):
        self.actions=[]
        self.states=[]
        self.rewards=[]
        self.terminal=[]
        self.logprobs=[]
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.terminal[:]
        del self.logprobs[:]
        
        
class Model(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Model,self).__init__()
        self.actor=nn.Sequential(
            nn.Linear(state_dim,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic=nn.Sequential(
            nn.Linear(state_dim,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self,state,action):
        action_prob=self.actor(state)
        dist=torch.distributions.Categorical(action_prob)
        logprobs=dist.log_prob(action)
        entropy=dist.entropy()
        values=self.critic(state)
        return logprobs,entropy,values
    def act(self,state):
        action_prob=self.actor(state)
        dist=torch.distributions.Categorical(action_prob)
        action=dist.sample()
        logprobs=dist.log_prob(action)
        return logprobs.detach(),action.detach()
    
class PPO:
    def __init__(self,gamma,actor_lr,critic_lr,clip_eps,update_epoch,state_dim,action_dim):
        self.gamma=gamma
        self.clip_eps=clip_eps
        self.update_epoch=update_epoch
        self.policy=Model(state_dim,action_dim).to(device)
        self.policy_old=Model(state_dim,action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer=Buffer()
        self.mse=nn.MSELoss()
        self.actor_opt=torch.optim.Adam(self.policy_old.actor.parameters(),actor_lr)
        self.critic_opt=torch.optim.Adam(self.policy_old.critic.parameters(),critic_lr)
    def select_action(self,state):
        state=torch.FloatTensor(state).to(device)
        with torch.no_grad():
            logprob,action=self.policy_old.act(state)
        self.buffer.logprobs.append(logprob)
        self.buffer.actions.append(action)
        self.buffer.states.append(state)
        return action.item()
    def update(self):
        rewards=[]
        sum_reward=0
        for reward,t in zip(reversed(self.buffer.rewards),reversed(self.buffer.terminal)):
            if t:
                sum_reward=0
            reward+=sum_reward*self.gamma
            rewards.insert(0,reward)
        rewards=torch.FloatTensor(rewards).to(device)
        rewards=(rewards-torch.mean(rewards))/(torch.std(rewards)+1e-8)
        old_states=torch.squeeze(torch.stack(self.buffer.states,dim=0)).detach().to(device)
        old_logprobs=torch.squeeze(torch.stack(self.buffer.logprobs,dim=0)).detach().to(device)
        old_acitons=torch.squeeze(torch.stack(self.buffer.actions,dim=0)).detach().to(device)
        for e in range(self.update_epoch):
            logprobs,entropy,values=self.policy(old_states,old_acitons)
            r=torch.exp(logprobs-old_logprobs)
            values=torch.squeeze(values)
            advantages=rewards-values.detach()
            arr1=r*advantages
            arr2=torch.clamp(r,1-self.clip_eps,1+self.clip_eps)*advantages
            loss=-torch.min(arr1,arr2)+0.5*self.mse(values,rewards)-0.05*entropy
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            loss.mean().backward()
            self.actor_opt.step()
            self.critic_opt.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    def save(self,path):
        torch.save(self.policy_old.state_dict(),path+'policy.pth')
            
            
    