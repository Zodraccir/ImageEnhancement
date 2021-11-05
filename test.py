import random,os,cv2
import gym
import numpy as np
import image_enhancement

if __name__ == '__main__':
    file = int(input("Please file :\n"))
    img_list = os.listdir("rawTest")[file:file+1]
    env = gym.make('image_enhancement-v0')
    for i in range(3):
        done = False

        #print(".......... EPISODE "+str(i)+" --------------")
        file=random.choice(img_list)

        img_path_raw = "rawTest/"+file
        print("img_path",img_path_raw)
        raw = cv2.imread(img_path_raw)
        img_path_exp = "ExpTest/"+file
        target = cv2.imread(img_path_exp)

        observation = env.reset(raw,target)

        action=1
        score=0

        print('initial distance', env.initial_distance)
        list_actions=[]
        while action:
            action = input("Please action a :\n")
            if(action == ""):
                break
            list_actions.append(int(action))
            observation_, reward, done, info = env.step(int(action))
            print(reward,info)
            score += reward
            lastinfo=info
        print('score: ', score)
        print('final distance ',env.initial_distance-lastinfo)
        env.doStepOriginal(list_actions)
        env.multiRender()


