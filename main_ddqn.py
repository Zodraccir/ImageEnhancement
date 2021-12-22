import argparse
import os
import random
import image_enhancement
import gym
import numpy as np

from ddqn_agent import DDQNAgent
from utils import plot_learning_curve

from torchvision import transforms
from PIL import Image

path_training_image="RawTraining/"
path_expert_image="ExpC/"
path_test_image="RawTest/"


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ngames', '-n', help="a number of games", type=int, default=1000)
    parser.add_argument('--epsdecay', '-e', help="epslon decay", type=float, default=2e-5)
    parser.add_argument('--learningRate','-lr', help="learningRate",type=float,default=0.001)
    parser.add_argument('--batchSize','-b', help="batch size",type=int,default=64)

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


    #lr=0002 RMSprop
    agent = DDQNAgent(gamma=0.80, epsilon=1.0, lr=args.learningRate,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=100000, eps_min=0.10,
                     batch_size=args.batchSize, replace=1000, eps_dec=args.epsdecay,
                     chkpt_dir='models/', algo='DDQNAgent',
                     env_name='image_enhancement-v0')

    if load_checkpoint:
        agent.load_models(learn_)

    n_steps = 0
    
    print('start execution, device used: ', agent.q_eval.device,' ,number games to execute: ',n_games, 'number action ',agent.n_actions,'learning rate: ',args.learningRate,' epslon decay: ',args.epsdecay , ' batch Size',args.batchSize)


    scores, eps_history, steps_array , scores_perc , numbers_actions = [], [], [], [], []
    img_list = os.listdir(path_training_image)



    '''
    file = random.choice(os.listdir("rawTest"))
    img_path_raw = "rawTest/" + file
    print("img_path", img_path_raw)
    raw = cv2.imread(img_path_raw)
    img_path_exp = "ExpTest/" + file
    target = cv2.imread(img_path_exp)
    '''
    convert_tensor = transforms.ToTensor()

    for i in range(n_games):
        done = False

        #print(".......... EPISODE "+str(i)+" --------------")
        file=random.choice(img_list)

        img_path_raw = Image.open(path_training_image+file)
        img_path_exp = Image.open(path_expert_image+file)

        raw=convert_tensor(img_path_raw)
        target=convert_tensor(img_path_exp)

        observation = env.reset(raw,target)
        state_= observation.detach().clone().to(agent.q_eval.device)
        score = 0

        n_actions=0
        final_distance=env.initial_distance
        initial_distance=env.initial_distance
        while not done:

            action = agent.choose_action(state_.unsqueeze_(0))
            #print("State_ mean: ",str(state_.mean())+ " std ",str(state_.std()) + "action done: ",action)
            observation_, reward, done, info = env.step(action)

            if done==11:
                break
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
            if not done:
            	final_distance = info





        score_perc=(1-(final_distance/initial_distance))*100

        steps_array.append(n_steps)
        numbers_actions.append(numbers_actions)
        scores.append(score)
        scores_perc.append(score_perc)
        eps_history.append(agent.epsilon)

        #avg_score = np.mean(scores[-100:])
        print('episode: ', i+1,'/',n_games,' Image:', file ,'score: %.1f' % score,
             ' percent score %.5f' % score_perc, ' number of actions ', n_actions,
            'epsilon %.2f' % agent.epsilon, 'initial distance', env.initial_distance ,'final distance' ,final_distance, 'steps', n_steps )



        #if load_checkpoint and n_steps >= 18000:
            #break

    if load_checkpoint:
    	agent.save_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    figure_file1= 'plots_custom/' + fname + 'custom.png'

    x = [i+1 for i in range(len(scores))]

    plot_learning_curve(x, scores, eps_history, figure_file1)
    plot_learning_curve(steps_array, scores, eps_history, figure_file)

    

