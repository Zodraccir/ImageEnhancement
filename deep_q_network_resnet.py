import torch as T
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary


class baseBlock(nn.Module):
    expansion=1
    def __init__(self,input_planes,planes,stride=1,dim_change=None):
        super(baseBlock,self).__init__()
        self.conv1=nn.Conv2d(input_planes,planes,stride=stride,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,stride=1,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(planes)
        self.dim_change=dim_change


    def forward(self,x):
        res=x
        output=F.relu(self.bn1(self.conv1(x)))
        output=self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res=self.dim_change(res)

        output+=res
        output=F.relu(output)

        return output


class bottleNeck(nn.Module):
    expansion=4

    def __init__(self,input_planes,planes,stride=1,dim_change=None):
        super(bottleNeck, self).__init__()

        self.conv1 =nn.Conv2d(input_planes,planes,kernel_size=1,stride=1)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv3=nn.Conv2d(planes,planes*self.expansion,kernel_size=1)
        self.bn3=nn.BatchNorm2d(planes*self.expansion)
        self.dim_change=dim_change

    def forward(self,x):
        res=x

        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))

        if self.dim_change is not None:
            res=self.dim_change(res)

        output+=res
        output=F.relu(output)
        return output

class DeepQNetwork(nn.Module):

    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir,block,num_layer):
        super(DeepQNetwork,self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)


        self.input_planes=64
        self.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(64)
        self.layer1=self._layer(block,64,num_layer[0],stride=1)
        self.layer2=self._layer(block,128,num_layer[1],stride=2)
        self.layer3=self._layer(block,256,num_layer[2],stride=2)
        self.layer4=self._layer(block,512,num_layer[3],stride=2)
        self.averagePool=nn.AvgPool2d(kernel_size=4,stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc=nn.Linear(fc_input_dims,n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _layer(self,block,planes,num_layer,stride=1):
        dim_change=None
        if stride!=1 or planes != self.input_planes*block.expansion:
            dim_change=nn.Sequential(nn.Conv2d(self.input_planes,planes*block.expansion,kernel_size=1,stride=stride),nn.BatchNorm2d(planes*block.expansion))

        netLayer=[]

        netLayer.append(block(self.input_planes,planes,stride=stride,dim_change=dim_change))
        self.input_planes=planes*block.expansion
        for i in range(1,num_layer):
            netLayer.append(block(self.input_planes,planes))
            self.input_planes=planes*block.expansion

        return nn.Sequential(*netLayer)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        x = F.relu(self.bn1(self.conv1(state)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        return int(np.prod(x.size()))

    def forward(self,x):

        x=F.relu(self.bn1(self.conv1(x)))

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=F.avg_pool2d(x,4)
        x=x.view(x.size(0),-1)
        x=self.fc(x)

        return x


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save({
        	'model_state_dict':self.state_dict(),
        	'optimizer_state_dict':self.optimizer.state_dict(),
        	'loss':self.loss
        	}
        	, self.checkpoint_file)

    def load_checkpoint(self,learn):
        print('... loading checkpoint ...')
        if(os.path.isfile(self.checkpoint_file)):
            checkpoint=T.load(self.checkpoint_file, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss=checkpoint['loss']
            if(learn):
                print("training mode")
                self.train()
            else:
                print("evaluation mode")
                self.eval()
        else:
        	print("file not exists")



'''
if __name__ =='__main__':
  # model = torchvision.models.AlexNet()

  input = T.randn(1, 3, 64, 64)


  model = DeepQNetwork(0.01,28,"",input.squeeze(0).shape,"",bottleNeck,[3,4,6,3])

  print(summary(model, (input.squeeze(0).shape)))

  input = T.randn(1,3,64,64)

  out = model(input)
'''