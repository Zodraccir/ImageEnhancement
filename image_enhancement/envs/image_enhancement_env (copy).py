import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torchvision.transforms as T
import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch
import image_enhancement.envs.actions as Act
import os,random



def calculateDistance(i1, i2):
	return torch.dist(i1,i2,2)
	
def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

class ImageEnhancementEnv(gym.Env):
	metadata = {'render.modes': ['human']}



	def __init__(self):
		self.action_space = spaces.Discrete(18)
		self.observation_space = spaces.Box(0, 255, [3, 256, 256])
		self.state = None
		self.previus_state= None
		self.target= None
		self.penality=None
		self.iteration=None
		self.initial_distance=None


	def step(self, action):
		assert self.action_space.contains(action)
		self.previus_state=self.state
		
		self.iteration+=1
		self.penality[action]+=1
		temp_state = self.state.unsqueeze_(0)

		if(action<6):

			act=action
			if(act<3):
				print("Action taken brightness positive in channel, ",int(act)," action",action)
				temp_state=Act.brightness(temp_state,0.1,int(act)).squeeze()
			else:
				act=act-3
				print("Action taken brightness negative in channel, ",int(act)," action",action)
				temp_state = Act.brightness(temp_state, -0.1,int(act)).squeeze()

		elif(action<12):
			act=action-6
			#print("*1", act)
			if (act < 3):
				print("Action taken gamma positive in channel, ",int(act)," action",action)
				temp_state = Act.gamma_corr(temp_state, 0.6, int(act)).squeeze()
			else:
				act = act - 3
				print("Action taken gamma negative in channel, ",int(act)," action",action)
				temp_state = Act.gamma_corr(temp_state, 1.1, int(act)).squeeze()

		elif(action<18):
			act=action-12
			#print("*1", act)
			if (act < 3):
				print("Action taken contrast positive in channel, ",int(act)," action",action)
				temp_state = Act.contrast(temp_state, 0.8, int(act)).squeeze()
			else:
				act = act - 3
				print("Action taken contrast negative in channel, ",int(act)," action",action)
				temp_state = Act.brightness(temp_state, 2, int(act)).squeeze()
		else:
			print(action)
		"""
		temp_state=torch.from_numpy(self.state)
		if action==0:
			temp_state=Act.brightness(temp_state,0.1)
		elif action==1:
			temp_state=Act.brightness(temp_state,0.8)
		elif action==2:
			temp_state=Act.gamma_corr(temp_state,0.6)
		elif action==3:
			temp_state=Act.brightness(temp_state,-0.1)
		elif action==4:
			temp_state=Act.contrast(temp_state,2)
		elif action==5:
			temp_state=Act.gamma_corr(temp_state,1.1)
		elif action == 6:
			a=temp_state.unsqueeze_(0)
			#print(a.shape)
			temp_state = Act.gamma_corr(a, -100.1,channel=2).squeeze()
			#print(temp_state.shape)
		else:
			print(action)

		"""
		#print("target", self.target)

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


		done=0
		#print("difference this state-targetstate",reward_state.item())
		if abs(reward_state.item())<threshold:
			done=1
			print("Passsaggi effettuati correttamente")

		if reward_state.item()>self.initial_distance+(self.initial_distance/2):
			done=1
			print("Limite sforato")
		if self.iteration>15:
			done=1

		#print(reward_state.item())
		#print(reward)

		return self.state.clone(), reward, done, self.penality


	def reset(self):




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

		self.iteration=0

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
