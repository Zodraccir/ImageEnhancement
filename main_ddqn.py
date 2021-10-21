import gym
import numpy as np
from ddqn_agent import DDQNAgent
import image_enhancement
from utils import plot_learning_curve, make_env
import sys
import argparse
import random,os,cv2



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ngames', '-n', help="a number of games", type=int, default=1000)
    parser.add_argument('--epsdecay', '-e', help="epslon decay", type=float, default=2e-5)

    #print(parser.format_help())
    # usage: test_args_4.py [-h] [--foo FOO] [--bar BAR]
    #
    # optional arguments:
    #      -h, --help         show this help message and exit
    #   --foo FOO, -f FOO  a random options
    #   --bar BAR, -b BAR  a more random option

    args = parser.parse_args()
    #print(args)  # Namespace(bar=0, foo='pouet')
    #print(args.ngames)  # pouet
    #print(args.epsdecay)  # 0

    env = gym.make('image_enhancement-v0')

  
    best_score = -np.inf
    load_checkpoint = True
    learn_= True
    n_games = args.ngames


    
    agent = DDQNAgent(gamma=0.99, epsilon=1.0, lr=0.00001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.05,
                     batch_size=64, replace=500, eps_dec=args.epsdecay,
                     chkpt_dir='models/', algo='DDQNAgent',
                     env_name='image_enhancement-v0')

    if load_checkpoint:
        agent.load_models(learn_)

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    
    print('start execution, device used: ', agent.q_eval.device,' ,number games to execute: ',n_games)
    
    scores, eps_history, steps_array = [], [], []

    img_list = os.listdir("rawTest")[:1]

    '''
    file = random.choice(os.listdir("rawTest"))
    img_path_raw = "rawTest/" + file
    print("img_path", img_path_raw)
    raw = cv2.imread(img_path_raw)
    img_path_exp = "ExpTest/" + file
    target = cv2.imread(img_path_exp)
    '''

    for i in range(n_games):
        done = False

        #print(".......... EPISODE "+str(i)+" --------------")
        file=random.choice(img_list)

        img_path_raw = "rawTest/"+file
        print("img_path",img_path_raw)
        raw = cv2.imread(img_path_raw)
        img_path_exp = "ExpTest/"+file
        target = cv2.imread(img_path_exp)

        observation = env.reset(raw,target)
        state_= observation.detach().clone().to(agent.q_eval.device)
        score = 0

        n_actions=0

        while not done:

            action = agent.choose_action(state_.unsqueeze_(0),n_actions)
            #print("State_ mean: ",str(state_.mean())+ " std ",str(state_.std()) + "action done: ",action)
            observation_, reward, done, info = env.step(action)
           
            
            #print("State +1 mean: ",str(observation_.mean())+ " std ",str(observation_.std()) + "reward done: ",reward)
            score += reward

            if learn_:
                agent.store_transition(state_.cpu(), action,
                                     reward, observation_, int(done))
                agent.learn()
            state_ = observation_.detach().clone()

            n_actions+=1
            #print("action " , n_actions, state_.sum())


            n_steps += 1
            #if done:
            	#print("finito")

        scores.append(score)  
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i+1,'/',n_games,'score: ', score,
             ' average score %.5f' % avg_score, 'best score %.5f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'initial distance', env.initial_distance , 'steps', n_steps )

        if(i>15):
            env.multiRender()

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
