import gym
import numpy as np
from ddqn_agent import DDQNAgent
import image_enhancement
from utils import plot_learning_curve, make_env

if __name__ == '__main__':
    env = gym.make('image_enhancement-v0')
    best_score = -np.inf
    load_checkpoint = True
    learn_= True
    n_games = 100000
    agent = DDQNAgent(gamma=0.99, epsilon=1.0, lr=0.001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=1000, eps_min=0.15,
                     batch_size=64, replace=500, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DDQNAgent',
                     env_name='image_enhancement-v0')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    
    print(agent.q_eval.device)
    
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        #print(".......... EPISODE "+str(i)+" --------------")
        observation = env.reset()
        state_= observation.clone().to(agent.q_eval.device)
        score = 0
        while not done:

            action = agent.choose_action(state_.unsqueeze_(0))
            #print("State_ mean: ",str(state_.mean())+ " std ",str(state_.std()) + "action done: ",action)
            observation_, reward, done, info = env.step(action)
           
            
            #print("State +1 mean: ",str(observation_.mean())+ " std ",str(observation_.std()) + "reward done: ",reward)
            score += reward

            if learn_:
                agent.store_transition(state_.cpu(), action,
                                     reward, observation_, int(done))
                agent.learn()
            state_ = observation_
            n_steps += 1
            #if done:
            	#print("finito")

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            #if not load_checkpoint:
            #    agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        #if load_checkpoint and n_steps >= 18000:
            #break

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
    
    if load_checkpoint:
    	agent.save_models()
