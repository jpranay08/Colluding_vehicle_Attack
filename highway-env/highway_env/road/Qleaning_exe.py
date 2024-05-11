
import gym
from qlearning import Agent
from utils import plot_learning_curve
import numpy as np
from customEnv import CustomEnv

if __name__ == '__main__':
    env =CustomEnv()
    agent= Agent(gamma=0.99, epsilon=1.0, batch_size =64, n_actions=100,eps_end =0.01, input_dims=[3], lr=0.003)
    scores, eps_history =[],[]
    n_games= 500

    for i in range(n_games):
        score=0
        done= False
        observation= env.reset()
        while not done:
            action=agent.choose_action(observation)

            observation_, reward, done , info =env.step(action,0.75,0.64)
            score+=reward
            agent.store_transition(observation, action, reward, observation_, done )
            agent.learn()
            observation= observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score =np.mean(scores[-100:])
        print('episode', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    x=[i+1 for i in range(n_games)]
    filename= 'lunar_lander-2020.png'
    plot_learning_curve(x, scores, eps_history, filename)

    