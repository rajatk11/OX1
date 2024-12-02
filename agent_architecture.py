import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        dropout = nn.Dropout(p=0.05)
        
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 40, (1,1), stride = 1, padding=(0,0)),
            #nn.Conv2d(40, 80, (1,2), stride = 2, padding=(0,1)),
            dropout,
            #nn.MaxPool2d((1, 2), stride = 1),
            nn.Conv2d(40, 80, (1,2), stride = 2, padding=(0,1)),
            nn.MaxPool2d((1, 1), stride = 1),
            nn.Conv2d(80, 100, (1,2), stride = 1, padding=(0,1)),
            #nn.MaxPool2d((1, 2), stride = 2),
            dropout,
            nn.Conv2d(100, 120, (1,2), stride = 1, padding=(0,1)),
            nn.MaxPool2d((1, 2), stride = 2),
            dropout,
            self.flatten,
            nn.Linear(1080,500),
            nn.ReLU(),
            dropout,
            #nn.Linear(6000,800),
            #nn.ReLU(),
            #dropout,
            nn.Linear(500, 60),
            nn.ReLU()
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(27, 27),
            nn.ReLU(),
            dropout,
            nn.Linear(27, 20),
            nn.ReLU(),
            dropout,
            nn.Linear(20, 12),
            nn.ReLU(),
            dropout,
            #nn.Linear(12, 7),
            #nn.ReLU(),
            dropout
        )
        self.aggr = nn.Sequential(
            nn.Linear(60+12, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            dropout,
            nn.Linear(20, 7),
            nn.ReLU()
            
        )

    def forward(self, x):
        
        
        #1
        logits2 = self.conv_stack(x[0])
        x1 = self.flatten(x[1])
        #x1 = x[1]
        logits1 = self.linear_relu_stack(x1)
        #print('Logits1 : ', logits1.size())
        #print('Logits2 : ', logits2.size())
        logits_cat = torch.cat((logits1, logits2), axis = 1)
        #print('Logitscat : ', logits_cat.size())
        logits = self.aggr(logits_cat)
       
        """
        
        #2
        logits = self.conv_stack(x)
        """
        
        """
        #3
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        """
        
        return logits
    
    
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        dropout = nn.Dropout(p=0.00)
        
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 40, (1,1), stride = 1, padding=(0,0)),
            #nn.Conv2d(40, 80, (1,2), stride = 2, padding=(0,1)),
            dropout,
            #nn.MaxPool2d((1, 2), stride = 1),
            nn.Conv2d(40, 80, (1,2), stride = 2, padding=(0,1)),
            nn.MaxPool2d((1, 1), stride = 1),
            nn.Conv2d(80, 100, (1,2), stride = 1, padding=(0,1)),
            #nn.MaxPool2d((1, 2), stride = 2),
            dropout,
            nn.Conv2d(100, 120, (1,2), stride = 1, padding=(0,1)),
            nn.MaxPool2d((1, 2), stride = 2),
            dropout,
            self.flatten,
            nn.Linear(1080,500),
            nn.ReLU(),
            dropout,
            #nn.Linear(6000,800),
            #nn.ReLU(),
            #dropout,
            nn.Linear(500, 60),
            nn.ReLU()
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(27, 27),
            nn.ReLU(),
            dropout,
            nn.Linear(27, 20),
            nn.ReLU(),
            dropout,
            nn.Linear(20, 12),
            nn.ReLU(),
            dropout,
            #nn.Linear(12, 7),
            #nn.ReLU(),
            dropout
        )
        self.aggr = nn.Sequential(
            nn.Linear(60+12, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            dropout,
            nn.Linear(20, 7),
            nn.ReLU()
            
        )
        
        self.act_select = nn.Sequential(
            nn.Linear(7 + 2, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU()
        )

    def forward(self, x):
        
        
        #1
        logits2 = self.conv_stack(x[0])
        x1 = self.flatten(x[1])
        #x1 = x[1]
        logits1 = self.linear_relu_stack(x1)
        #print('Logits1 : ', logits1.size())
        #print('Logits2 : ', logits2.size())
        logits_cat = torch.cat((logits1, logits2), axis = 1)
        #print('Logitscat : ', logits_cat.size())
        logits2 = self.aggr(logits_cat)
        logits_cat1 = torch.cat((logits2,x[2]), axis = 1)
        
       
        """
        
        #2
        logits = self.conv_stack(x)
        """
        
        """
        #3
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        """
        
        return self.act_select(logits_cat1)