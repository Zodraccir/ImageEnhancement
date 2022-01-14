import random,os,cv2
import gym
import numpy as np
import image_enhancement
from PIL import Image
from torchvision import transforms


if __name__ == '__main__':
    file = int(input("Please file :\n"))
    img_list = os.listdir("RawTest")[file:file+1]
    env = gym.make('image_enhancement-v0')

    convert_tensor = transforms.ToTensor()

    for i in range(3):
        done = False

        #print(".......... EPISODE "+str(i)+" --------------")

        #file = random.choice(img_list)

        file="1964.png"

        img_path_raw = Image.open("RawTest/" + file)
        img_path_exp = Image.open("ExpC/" + file)

        raw = convert_tensor(img_path_raw)
        target = convert_tensor(img_path_exp)

        observation = env.reset(raw, target)

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


