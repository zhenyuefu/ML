from collections import defaultdict
import gym
import flappy_bird_gym
import time
import numpy as np
from collections import defaultdict



class MyFlappyEnv:
    """ Custom Flappy Env :
        * state : [horizontal delta of the next pipe, vertical delta, vertical velocity]
    """

    def __init__(self):
        self.env = flappy_bird_gym.make('FlappyBird-v0')
        self.env._normalize_obs = False
        self._last_score = 0
    def __getattr__(self,attr):
        return self.env.__getattribute__(attr)
    
    def step(self,action):
        obs, reward, done, info = self.env.step(action)
        if done:
            reward -=1000
        player_x = self.env._game.player_x
        player_y = self.env._game.player_y

        return np.hstack([obs,self.env._game.player_vel_y]),reward, done, info
    def reset(self):
        return np.hstack([self.env.reset(),self.env._game.player_vel_y])

def test_gym(fps=30):
    env = gym.make('Taxi-v3')
    env.reset()
    r = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        r += reward
        env.render()
        time.sleep(1/fps)
        print(f"iter {i} : action {action}, reward {reward}, state {type(obs)} ")
        if done:
            break
    print(f"reward cumulatif : {r} ")
 


def test_flappy(fps=30):
    env = flappy_bird_gym.make('FlappyBird-v0')
    env.reset()
    r = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        r += reward
        env.render()
        time.sleep(1/fps)
        print(f"iter {i} : action {action}, reward {reward}, state {obs} {info}, {env._game.player_vel_y}")
        if done:
            break
    print(f"reward cumulatif : {r} ")
 


def play_env(agent,max_ep=500,fps=-1,verbose=False):
    """
        Play an episode :
        * agent : agent with two functions : act(state) -> action, and store(state,action,state,reward)
        * max_ep : maximal length of the episode
        * fps : frame per second,not rendering if <=0
        * verbose : True/False print debug messages
        * return the cumulative reward
    """
    obs = agent.env.reset()
    cumr = 0
    for i in range(max_ep):
        last_obs = obs
        action = agent.act(obs)
        obs,reward,done,info = agent.env.step(int(action))
        agent.store(last_obs,action,obs,reward)
        cumr += reward
        if fps>0:
            agent.env.render()
            if verbose: print(f"iter {i} : {action}: {reward} -> {obs} ")        
            time.sleep(1/fps)
        if done:
            break
    return cumr


class AgentRandom:
    """
         A simple random agent
    """
    def __init__(self,env):
        self.env = env
    def act(self,obs):
        return self.env.action_space.sample()
    def store(self,obs,action,new_obs,reward):
        pass


class AgentPolicy:
    """
        Agent following a policy pi : pi is a dictionary state -> action
    """
    def __init__(self,env,pi):
        self.env = env
        self.pi = pi
    def act(self,obs):
        return self.pi[obs]
    def store(self,obs,action,new_obs,reward):
        pass

if __name__ == "__main__":
    test_flappy()
    test_gym()
    envTaxi = gym.make('Taxi-v3')
    envFlappy = flappy_bird_gym.make('FlappyBird-v0')
    play_env(AgentRandom(envTaxi),fps=30)
    play_env(AgentRandom(envFlappy),fps=60)