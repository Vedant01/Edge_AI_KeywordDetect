'''
date: 2023-05-08
author: Renyuan Lyu

modified by Joseph Lin

'''

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import collections

theLabels= [
 'backward', 'bed',     'bird',     'cat',      'dog',
 'down',    'eight',    'five',     'follow',   'forward',
 'four',    'go',       'happy',    'house',    'learn',
 'left',    'marvin',   'nine',     'no',       'off',
 'on',      'one',      'right',    'seven',    'sheila',
 'six',     'stop',     'three',    'tree',     'two',
 'up',      'visual',   'wow',      'yes',      'zero'
]

def label_to_index(label):
    return torch.tensor(theLabels.index(label))

def index_to_label(index):
    return theLabels[index]
#%%
# get the number of parameters in the model
def get_n_params(model):
    np= 0
    for p in model.parameters():
        np += p.numel()
    return np

ryMelsgram= torchaudio.transforms.MelSpectrogram(
    sample_rate= 16_000,   # 16 kHz
    hop_length=  160,      # 10 ms
    n_fft=       160*2,    # 20 ms
    n_mels=      64, 
)
class ryMelsgram1d(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x= ryMelsgram(x)
        x= x.squeeze(dim=-3) 
        return x

class ryMelsgram2d(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x= ryMelsgram(x)   
        #x= x.unsqueeze(dim=-3) # add channel dim，可能不需要
        return x
class ryAvgPool1d(nn.Module):
    def __init__(self, o_size= 1):
        super().__init__()
        self.pool= nn.AdaptiveAvgPool1d(o_size) # output size= 1
    def forward(self, x):
        x= self.pool(x)
        x= x.squeeze(dim=-1)
        return x

class ryAvgPool2d(nn.Module):
    def __init__(self, o_size= 1):
        super().__init__()
        self.pool= nn.AdaptiveAvgPool2d(o_size) # output size= 1
    def forward(self, x):
        x= self.pool(x)
        x= x.squeeze(dim=(-2,-1))
        return x
device= torch.device('cuda' if torch.cuda.is_available() else 
                     'cpu')
ryMelsgram= ryMelsgram.to(device)

    
# check the availability of GPU


#%%








#%%
xB= torch.randn((10, 16_000))
xB= xB.cuda()
X_mt= melsgram= \
torchaudio\
.transforms\
.MelSpectrogram().to('cuda')(xB)
X_mt.shape # torch.Size([10, 128, 81])
# %%

class ryM1(nn.Module):
    
    def __init__(self, 
                 in_chs=   1,  #  1 channel, mono waveform
                 out_cls= 35,  # 35 words as output classes
                 sample_rate=  16_000 # sample rate of the audio file
                 ):
        
        super().__init__()

        new_sample_rate= sample_rate #//2 #8_000

        n_fft=  int(.02*sample_rate) #320
        n_mels=  64
        self.transform= torchaudio.transforms.MelSpectrogram(
            sample_rate= sample_rate, # 16,000 # 1000ms
            n_fft= n_fft,             # 320    # 20ms
            n_mels= n_mels)           # 64       


        self.act=  nn.ReLU()
        self.flat= nn.Flatten()
        self.out=  nn.LogSoftmax(dim=-1)
      
        '''
        k1= int(.02 *new_sample_rate) # 320 # 20ms
        s1= int(.01 *new_sample_rate) # 160 # 10ms
        ch1= 64 # 64 channels in 1st convolution layer
        '''
        k1= n_fft
        s1= k1//2
        ch1= n_mels #64 # 64 channels in 1st convolution layer

        k2= 4 # kernel size in the other conv layer
        s2= 2 # stride in the other conv layer

        self.conv2= nn.Conv1d(64,  128,  kernel_size= 4, stride= 2)
        self.bn2=   nn.BatchNorm1d(128)

        self.conv3= nn.Conv1d(128, 256, kernel_size= 4, stride= 2)
        self.bn3=   nn.BatchNorm1d(256)

        self.conv4= nn.Conv1d(256, 256, kernel_size= 4, stride= 2)
        self.bn4=   nn.BatchNorm1d(256)

        self.conv5= nn.Conv1d(256, 128, kernel_size= 4, stride= 2)
        self.bn5=   nn.BatchNorm1d(128)
        
        self.fc1= nn.Linear(128, 64)
        self.fc2= nn.Linear(64,35) #35 outputs

    def forward(self, x):
        
        #x= self.transform(x) # (1,16000) -> (1,8000) # downsample by factor of 2

        #  CNNs
        x= self.transform(x) #  -> (, 1, 64, 101)
        x= x.squeeze(1) # -> (, 64, 101)
        #x= self.bn1(x)   
        #x= self.act(x)   
        
        x= self.conv2(x) #  -> (,128, 49)
        x= self.bn2(x)   
        x= self.act(x)   
        
        x= self.conv3(x) #  -> (,256, 23)
        x= self.bn3(x)   
        x= self.act(x)   
       
        x= self.conv4(x) #  -> (,256, 10)
        x= self.bn4(x)   
        x= self.act(x)

        x= self.conv5(x) #  -> (,128, 4)
        x= self.bn5(x)   
        x= self.act(x)   
        
        # global average pooling
        x= F.avg_pool1d(x, x.shape[-1])  # -> (,128, 1)
        x= self.flat(x) # -> (,128)

        # MLPs
        x= self.fc1(x)  # -> (,64)
        x= self.act(x)

        x= self.fc2(x)  # -> (,35)
        y= self.out(x)  # -> (,35)

        return y

##
## MelSpectrogram + CNNs + MLPs
## 1. 1st conv layer: 64 channels, kernel size= 320, stride= 160
## 2. 2nd conv layer: 128 channels, kernel size= 4, stride= 2
## 3. 3rd conv layer: 256 channels, kernel size= 4, stride= 2
## 4. 4th conv layer: 256 channels, kernel size= 4, stride= 2
## 5. 5th conv layer: 128 channels, kernel size= 4, stride= 2
## 6. 1st fc layer: 64 neurons
## 7. 2nd fc layer: 35 neurons
## 8. output layer: 35 neurons
##


class ryM2(nn.Module):
    def __init__(self, in_chs= 1, out_cls= 35):
        super(ryM2, self).__init__()
        
        layers= collections.OrderedDict([

            #('c1', nn.Conv1d(in_chs,  64, kernel_size= 320, stride= 160)),
            #('b1', nn.BatchNorm1d(64)),
            #('r1', nn.ReLU()),
            ('mels', ryMelsgram1d()),
            
            ('c2', nn.Conv1d(64,  128, kernel_size= 4, stride= 2)),
            ('b2', nn.BatchNorm1d(128)),
            ('r2', nn.ReLU()),
            
            ('c3', nn.Conv1d(128, 256, kernel_size= 4, stride= 2)),
            ('b3', nn.BatchNorm1d(256)),
            ('r3', nn.ReLU()),
            
            ('c4', nn.Conv1d(256, 256, kernel_size= 4, stride= 2)),
            ('b4', nn.BatchNorm1d(256)),
            ('r4', nn.ReLU()),
            
            ('c5', nn.Conv1d(256, 256, kernel_size= 4, stride= 2)),
            ('b5', nn.BatchNorm1d(256)),
            ('r5', nn.ReLU()),
            
            ('p1', ryAvgPool1d()), 
            
            ('l1', nn.Linear(256, 128)),
            ('t1', nn.Tanh()),
            
            ('l2', nn.Linear(128, out_cls)),
            ('out',nn.LogSoftmax(dim=-1))
            ])
        
        self.model= nn.Sequential(layers)

    def forward(self, x):
        x= self.model(x)
        return x


# a torch layer to average the output of the last layer




class ryM3(nn.Module):
    def __init__(self, 
                 in_chs=   1,  #  1 channel, mono waveform
                 out_cls= 35,  # 35 words as output classes
                 sample_rate=  16_000 # sample rate of the audio file
                 ):
        
        super().__init__()

        new_sample_rate= sample_rate #//2 #8_000

        #self.transform= torchaudio.transforms.Resample(
        #    orig_freq= sample_rate, 
        #    new_freq=  new_sample_rate)

        self.act=  nn.ReLU()
        self.flat= nn.Flatten()
        self.out=  nn.LogSoftmax(dim=-1)
        #self.out=  nn.Softmax(dim=-1)

        k1= int(.02 *new_sample_rate) # 320 # 20ms
        s1= int(.01 *new_sample_rate) # 160 # 10ms
        ch1= 64 # 64 channels in 1st convolution layer

        k2= 4 # kernel size in the other conv layer
        s2= 2 # stride in the other conv layer

        self.conv1= nn.Conv1d(1, 64,   kernel_size= 320, stride= 160) 
        self.bn1=   nn.BatchNorm1d(64)

        self.conv2= nn.Conv1d(64,  128,  kernel_size= 4, stride= 2)
        self.bn2=   nn.BatchNorm1d(128)

        self.conv3= nn.Conv1d(128, 256, kernel_size= 4, stride= 2)
        self.bn3=   nn.BatchNorm1d(256)

        self.conv4= nn.Conv1d(256, 256, kernel_size= 4, stride= 2)
        self.bn4=   nn.BatchNorm1d(256)

        self.conv5= nn.Conv1d(256, 128, kernel_size= 4, stride= 2)
        self.bn5=   nn.BatchNorm1d(128)
        
        self.fc1= nn.Linear(128, 65)
        self.fc2= nn.Linear(65,35)

    def forward(self, x):
        
        #x= self.transform(x) # (1,16000) -> (1,8000) # downsample by factor of 2

        #  CNNs
        x= self.conv1(x) #  -> ( 64, 99)
        x= self.bn1(x)   
        x= self.act(x)   
        
        x= self.conv2(x) #  -> (128, 48)
        x= self.bn2(x)   
        x= self.act(x)   
        
        x= self.conv3(x) #  -> (256, 23)
        x= self.bn3(x)   
        x= self.act(x)   
       
        x= self.conv4(x) #  -> (256, 10)
        x= self.bn4(x)   
        x= self.act(x)

        x= self.conv5(x) #  -> (128, 4)
        x= self.bn5(x)   
        x= self.act(x)   
        
        # global average pooling
        x= F.avg_pool1d(x, x.shape[-1])  # -> (128, 1)
        x= self.flat(x) # -> (128)

        # MLPs
        x= self.fc1(x)  # -> (64)
        x= self.act(x)

        x= self.fc2(x)  # -> (35)
        y= self.out(x)  # -> (35)

        return y

# raw waveform -> CNNs -> MLPs -> output
# 1D convolutional neural network

# Number of parameters: 590_563

class ryM4(nn.Module):
    def __init__(self, in_chs= 1, out_cls= 35):
        super(ryM4, self).__init__()
        
        layers= collections.OrderedDict([

            ('c1', nn.Conv1d(in_chs,  64, kernel_size= 320, stride= 160)),
            ('b1', nn.BatchNorm1d(64)),
            ('r1', nn.ReLU()),
            #('mels', ryMelsgram1d()),
            
            ('c2', nn.Conv1d(64,  128, kernel_size= 4, stride= 2)),
            ('b2', nn.BatchNorm1d(128)),
            ('r2', nn.ReLU()),
            
            ('c3', nn.Conv1d(128, 256, kernel_size= 4, stride= 2)),
            ('b3', nn.BatchNorm1d(256)),
            ('r3', nn.ReLU()),
            
            ('c4', nn.Conv1d(256, 256, kernel_size= 4, stride= 2)),
            ('b4', nn.BatchNorm1d(256)),
            ('r4', nn.ReLU()),
            
            ('c5', nn.Conv1d(256, 256, kernel_size= 4, stride= 2)),
            ('b5', nn.BatchNorm1d(256)),
            ('r5', nn.ReLU()),
            
            ('p1', ryAvgPool1d()), 
            
            ('l1', nn.Linear(256, 128)),
            ('t1', nn.Tanh()),
            
            ('l2', nn.Linear(128, out_cls)),
            ('out',nn.LogSoftmax(dim=-1))
            ])
        
        self.model= nn.Sequential(layers)

    def forward(self, x):
        x= self.model(x)
        return x

#
# Simply use cov1d to replace melspectrogram




