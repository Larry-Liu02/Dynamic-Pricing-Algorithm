from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, count = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), count 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(agent,num_episodes,xi,c):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'counts': []}
                random.seed(1)
#                 state = [random.uniform(5,10),random.randint(400,500)]
                state = [35,500]
                next_state =[0,0]
                count = 0
                while count < 100:
                    action = agent.take_action(state)
                    next_state[0]= state[0] + action[0]
                    next_state[1] =max(0,-(action[0] * state[1] * xi / state[0]) + state[1])
                    reward = (next_state[0] - c) * next_state[1]
                    count += 1
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['counts'].append(count)
                    state = next_state
                    next_state =[0,0]
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(agent, num_episodes, replay_buffer, minimal_size, batch_size,xi,c):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                random.seed(1)
#                 state = [random.uniform(5,10),random.randint(400,500)]
                state = [35,500]
                next_state =[0,0]
                count = 0
                while count < 100:
                    action = agent.take_action(state)
                    next_state[0]= state[0] + action[0]
                    next_state[1] = -(action[0] * state[1] * xi / state[0]) + state[1]
                    reward = (next_state[0] - c) * next_state[1]
                    count += 1
                    replay_buffer.add(state, action, reward, next_state, count)
                    state = next_state
                    next_state =[0,0]
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_c = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'counts': b_c}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def human_agent_05(num_episodes,xi,c):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = [35,500]
                next_state =[0,0]
                count = 0
                while count < 100:
                    if count < 90:
                        action = [2]
                        next_state[0]= state[0] + action[0]
                        next_state[1] = -(action[0] * state[1] * xi / state[0]) + state[1]
                        reward = (next_state[0] - c) * next_state[1]
                        count += 1
                        state = next_state
                        next_state =[0,0]
                        episode_return += reward
                        
                    else:
                        action = [0]
                        next_state[0]= state[0] + action[0]
                        next_state[1] = -(action[0] * state[1] * xi / state[0]) + state[1]
                        reward = (next_state[0] - c) * next_state[1]
                        count += 1
                        next_state =[0,0]
                        episode_return += reward                   
                        
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def human_agent_2(num_episodes,xi,c):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = [35,500]
                next_state =[0,0]
                count = 0
                while count < 100:
                    if count < 5:
                        action = [2]
                        next_state[0]= state[0] + action[0]
                        next_state[1] = -(action[0] * state[1] * xi / state[0]) + state[1]
                        reward = (next_state[0] - c) * next_state[1]
                        count += 1
                        state = next_state
                        next_state =[0,0]
                        episode_return += reward
                        
                    else:
                        action = [0]
                        next_state[0]= state[0] + action[0]
                        next_state[1] = -(action[0] * state[1] * xi / state[0]) + state[1]
                        reward = (next_state[0] - c) * next_state[1]
                        count += 1
                        next_state =[0,0]
                        episode_return += reward                   
                        
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def human_agent_5(num_episodes,xi,c):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = [35,500]
                next_state =[0,0]
                count = 0
                while count < 100:
                    if count < 3:
                        action = [2]
                        next_state[0]= state[0] + action[0]
                        next_state[1] = -(action[0] * state[1] * xi / state[0]) + state[1]
                        reward = (next_state[0] - c) * next_state[1]
                        count += 1
                        state = next_state
                        next_state =[0,0]
                        episode_return += reward
                        
                    else:
                        action = [0]
                        next_state[0]= state[0] + action[0]
                        next_state[1] = -(action[0] * state[1] * xi / state[0]) + state[1]
                        reward = (next_state[0] - c) * next_state[1]
                        count += 1
                        next_state =[0,0]
                        episode_return += reward                   
                        
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list
    


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                