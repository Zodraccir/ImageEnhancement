import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    ''' first model
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)

        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions
    '''
    '''second model
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 96, 11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, 4, stride=1,padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1,padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1,padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, stride=1,padding=1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        print(fc_input_dims)
        self.fc1 = nn.Linear(fc_input_dims, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        print("1",state.shape)
        dims = self.conv1(state)
        print("2",dims.shape)
        dims = self.maxpool1(dims)
        print("2pool", dims.shape)
        dims = self.conv2(dims)
        print("3",dims.shape)
        dims = self.maxpool2(dims)
        print("3pool", dims.shape)
        dims = self.conv3(dims)
        print("4",dims.shape)
        dims = self.conv4(dims)
        print("5", dims.shape)
        dims = self.conv5(dims)
        print("6", dims.shape)
        print(dims.view(dims.size()[0],-1).shape)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = self.maxpool1(F.relu(self.conv1(state)))
        conv2 = self.maxpool2(F.relu(self.conv2(conv1)))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        conv5 = F.relu(self.conv5(conv4))
        conv_state = conv5.view(conv5.size()[0], -1)
        #print("conv",conv_state.shape)
        flat1 = F.relu(self.fc1(conv_state))
        flat2 = F.relu(self.fc2(flat1))
        flat3 = F.relu(self.fc3(flat2))
        actions = self.fc4(flat3)

        return actions
    '''

    '''adam model'''
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.features=nn.Sequential(
            nn.Conv2d(input_dims[0], 64, 3, 2, 1),  # in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2),  # kernel_size
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_input_dims, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_actions),
        )

        #self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims=self.features(state)
        #print(dims.view(dims.size()[0], -1).shape)
        return int(np.prod(dims.size()))

    def forward(self,state):
        x=self.features(state)
        #print(x.shape)
        h=x.view(x.shape[0],-1)
        #print(h.shape)
        actions=self.classifier(h)
        return actions


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
