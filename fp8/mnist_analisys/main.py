import torch.optim as optim
from mnist_analisys.adam import ScaledAdam
import torch.nn as nn
from model import Model
from torch.utils.tensorboard import SummaryWriter
from mnist_analisys.config import Config as config
from mnist_analisys.learningProcess import LearningProcess
import time
import torch

torch.manual_seed(20)

if __name__ == "__main__":
    model = Model()
    writer = SummaryWriter(config.pathToLogs + "_" + str(time.time()))
    
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    optimizer = ScaledAdam(model.parameters(), lr=config.learning_rate, writer=writer)
    criterion = nn.CrossEntropyLoss()
    learner = LearningProcess(optimizer, criterion, writer)
    learner.train(model)
    # learner.validate(model)
    # learner.test(model)
