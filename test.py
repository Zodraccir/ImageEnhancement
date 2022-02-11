import random,os,cv2
import gym
import numpy as np
import image_enhancement
from PIL import Image
from torchvision import transforms
from scipy.special import softmax


if __name__ == '__main__':
    '''
    distances=[34,9,3,4,44,1]



    keys=list(range(0, len(distances)))
    x=dict(zip(keys,distances))


    print (x)
    x= {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
    max = max(x.values())  # max value in sense of minimum distance from targer
    min = min(x.values())  # min value in sense of maximum distance from targer

    #print(max,min)
    print(x)
    for i in range(0,len(distances)):
        value=(2*(x[i]-max)/(min-max))-1
        if value>0:
            value=pow(value,3)
        x[i]=value


    print (x)

    print(x[0])

    '''
    file = int(input("Please file :\n"))
    img_list = os.listdir("RawTest")[file:file+1]
    env = gym.make('image_enhancement-v0')

    convert_tensor = transforms.ToTensor()

    for i in range(3):
        done = False

        #print(".......... EPISODE "+str(i)+" --------------")

        #file = random.choice(img_list)

        file="0004.png"

        img_path_raw = Image.open("RawTraining/" + file)
        img_path_exp = Image.open("ExpC/" + file)

        raw = convert_tensor(img_path_raw)
        target = convert_tensor(img_path_exp)

        observation = env.reset(raw, target)

        action=1
        score=0

        print('initial distance', env.initial_distance)
        list_actions=[]
        while not done:
            action = input("Please action a :\n")

            list_actions.append(int(action))
            observation_, reward, done, info = env.step(int(action))
            print(reward,info)
            score += reward
            lastinfo=info

        print('score: ', score)
        print('final distance ',lastinfo)
        env.doStepOriginal(list_actions)
        env.multiRender()


