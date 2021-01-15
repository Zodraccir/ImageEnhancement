import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torchvision.transforms as T
import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch


import os,random


class Act():
    def gamma_corr(image, gamma, channel=None):
        mod = image.clone()
        if channel is not None:
            mod[:, channel, :, :] = mod[:, channel, :, :] ** gamma
        else:
            mod = mod ** gamma
        return mod

    def brightness(image, bright, channel=None):
        mod = image.clone()
        if channel is not None:
            mod[:, channel, :, :] = torch.clamp(mod[:, channel, :, :] + bright, 0, 1)
        else:
            mod = torch.clamp(mod + bright, 0, 1)

        return mod

    def contrast(image, alpha, channel=None):
        mod = image.clone()
        if channel is not None:
            mod[:, channel, :, :] = torch.clamp(
                torch.mean(mod[:, channel, :, :]) + alpha * (mod[:, channel, :, :] - torch.mean(mod[:, channel, :, :])),
                0, 1)
        else:
            mod = torch.clamp(torch.mean(mod) + alpha * (mod - torch.mean(mod)), 0, 1)
        return mod
	

def calculateDistance(i1, i2):
	return torch.dist(i1,i2,2)
	
def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

class ImageEnhancementEnv(gym.Env):
	metadata = {'render.modes': ['human']}



	def __init__(self):
		self.action_space = spaces.Discrete(6)
		self.observation_space = spaces.Box(0, 255, [3, 256, 256])
		self.state = None
		self.previus_state= None
		self.target= None
		self.penality=None
		self.iteration=0
		self.initial_distance=None


	def step(self, action):
		assert self.action_space.contains(action)
		self.previus_state=self.state
		

		self.penality[action]+=1
		temp_state = self.state.unsqueeze_(0)
		#print(action)
		self.iteration+=1
		
		if(action<3):

			act=action
			if(act==0):
				#print("Action taken brightness positive, action",action)
				temp_state=Act.brightness(temp_state,0.08,0).squeeze()
			if(act==1):
				temp_state=Act.brightness(temp_state,0.08,1).squeeze()
			else:
				
				#print("Action taken brightness negative in channel, action",action)
				temp_state = Act.brightness(temp_state, 0.08,2).squeeze()
		elif(action<6):
			
			#print("*1", act)
			act=action-3
			if(act==0):
				temp_state=Act.brightness(temp_state,-0.08,0).squeeze()
			if (act==1):
				#print("Action taken contrast positive in channel,  action",action)
				temp_state=Act.brightness(temp_state,-0.08,1).squeeze()
			else:
				#print("Action taken contrast negative in channel, action",action)
				temp_state=Act.brightness(temp_state,-0.08,2).squeeze()
		else:
			print(action)

		
		self.state=temp_state

		#print("action",self.state)
		reward_state=mse_loss(self.target,self.state)
		reward = mse_loss(self.target,self.previus_state)-reward_state
		#print(reward_state)

		threshold=2.0
		max_threshold=-120
		#print(reward)

		#reward=2*reward

		#if(self.penality[action]>1):
			#reward_state=reward_state- self.penality[action]*50
			#reward=reward-self.penality[action]*100

		#reward=reward-self.iteration*15

		#print(reward_state.item())

		done=0
		#print("difference this state-targetstate",reward_state.item())
		if abs(reward_state.item())<threshold:
			done=1
			print("Passsaggi effettuati correttamente")

		if reward_state.item()>self.initial_distance+(self.initial_distance/2):
			done=1
			#print("Limite sforato")
		if self.iteration>15:
			done=1
			 #print("Max operazioni effettuate")

		#print(reward_state.item())
		#print(reward)

		return self.state.clone(), reward, done, self.penality


	def reset(self):


		
		self.iteration=0
		file=random.choice(os.listdir("rawTest"))
		#file=os.listdir("rawTest")[m]
		img_path = "rawTest/"+file
		
		
		
		print("img_path",img_path)
		#img_path="img.png"
		#print(file)

		transform = T.Compose([T.ToTensor()])
		img = cv2.imread(img_path)
		height, width, channels = img.shape

		#print(height, width, channels)
		img1 = transform(img)
		#im2 = img1.numpy()
		# print(im2)
		# im2=im2.astype('int8')
		#im2 = np.transpose(im2, (1, 2, 0))

		img_path1 = "ExpTest/"+file
		#img_path1="final.png"

		transform = T.Compose([T.ToTensor()])
		im = cv2.imread(img_path1)
		im1 = transform(im)
		#i2 = im1.numpy()
		# print(im2)
		# im2=im2.astype('int8')
		#i2 = np.transpose(i2, (1, 2, 0))
		
		self.target=im1

		self.state=img1

		self.penality=dict.fromkeys(range(self.action_space.n), 0)

		#print (self.target)


		self.initial_distance=mse_loss(self.target,self.state)


		return self.state

	def render(self):
		rdner=np.transpose(self.state.numpy(),(1,2,0))
		plt.imshow(cv2.cvtColor(rdner, cv2.COLOR_BGR2RGB))
		plt.show()


	def save(self,name):
		#print(self.state)
		rdner = np.transpose(self.state.numpy(), (1, 2, 0))
		matplotlib.image.imsave(name+'.png', (cv2.cvtColor(rdner, cv2.COLOR_BGR2RGB)))
		#cv2.imwrite("final.png",cv2.cvtColor(self.state, cv2.COLOR_BGR2RGB))
