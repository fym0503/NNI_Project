import time
import math
import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from model import smallVGG
def setLearningRate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def train(DATASET = 'CIFAR10', NEpochs = 200,
        BatchSize = 32, learningRate = 1e-3, NDecreaseLR = 20, eps = 1e-3,
        AMSGrad = True, model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth", gpu = True):

    print("DATASET: ", DATASET)


    mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
    Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
    tsf = {
        'train': transforms.Compose(
        [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees = 10, translate = [0.1, 0.1], scale = [0.9, 1.1]),
        transforms.ToTensor(),
        Normalize
        ]),
       'test': transforms.Compose(
        [
        transforms.ToTensor(),
        Normalize
        ])
        }
    trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = True,
                                        download=True, transform = tsf['train'])
    testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = False,
                                       download=True, transform = tsf['test'])

    net = smallVGG(3)

    x_train, y_train = trainset.data, trainset.targets,
    x_test, y_test = testset.data, testset.targets,

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BatchSize,
                                      shuffle = True, num_workers = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1000,
                                      shuffle = False, num_workers = 1)

    trainIter = iter(trainloader)
    testIter = iter(testloader)
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    if gpu:
        net.cuda()
        criterion.cuda()
        softmax.cuda()

    optimizer = optim.Adam(params = net.parameters(), lr = learningRate, eps = eps, amsgrad = AMSGrad)

    NBatch = int(len(trainset) / BatchSize)
    cudnn.benchmark = True
    for epoch in range(NEpochs):
        lossTrain = 0.0
        accTrain = 0.0
        for i in range(NBatch):
            try:
                batchX, batchY = trainIter.next()
            except StopIteration:
                trainIter = iter(trainloader)
                batchX, batchY = trainIter.next()

            if gpu:
                batchX = batchX.cuda()
                batchY = batchY.cuda()

            optimizer.zero_grad()
            logits = net.forward(batchX)
            prob = softmax(logits)

            loss = criterion(logits, batchY)
            loss.backward()
            optimizer.step()

            lossTrain += loss.cpu().detach().numpy() / NBatch
            if gpu:
                pred = np.argmax(prob.cpu().detach().numpy(), axis = 1)
                groundTruth = batchY.cpu().detach().numpy()
            else:
                pred = np.argmax(prob.detach().numpy(), axis = 1)
                groundTruth = batchY.detach().numpy()

            acc = np.mean(pred == groundTruth)
            accTrain += acc / NBatch

        if (epoch + 1) % NDecreaseLR == 0:
            learningRate = learningRate / 2.0
            setLearningRate(optimizer, learningRate)
        accTest,lossTest = test(testloader, net, gpu=gpu)
        print("Epoch: ", epoch, "Train Loss: ", round(lossTrain,4), "Train accuracy: ", round(accTrain*100,2), "Test Loss: ",\
              round(lossTest,4),"Test accuracy: ",round(accTest*100,2))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(net, model_dir + model_name)
    print("Model saved")
def test(testloader, net,gpu = True):
    testIter = iter(testloader)
    acc = 0.0
    NBatch = 10
    cri = nn.CrossEntropyLoss()
    lossTest = 0.0
    for i, data in enumerate(testIter, 0):
        batchX, batchY = data
        if gpu:
            batchX = batchX.cuda()
            batchY = batchY.cuda()
        logits = net.forward(batchX)
        loss = cri(logits, batchY)
        lossTest += loss.cpu().detach().numpy() / NBatch
        if gpu:
            pred = np.argmax(logits.cpu().detach().numpy(), axis = 1)
            groundTruth = batchY.cpu().detach().numpy()
        else:
            pred = np.argmax(logits.detach().numpy(), axis = 1)
            groundTruth = batchY.detach().numpy()
        acc += np.mean(pred == groundTruth)
    accTest = acc / NBatch
    return accTest,lossTest
if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'CIFAR10')
        parser.add_argument('--epochs', type = int, default = 100)
        parser.add_argument('--eps', type = float, default = 1e-3)
        parser.add_argument('--AMSGrad', type = bool, default = True)
        parser.add_argument('--batch_size', type = int, default = 32)
        parser.add_argument('--learning_rate', type = float, default = 1e-3)
        parser.add_argument('--decrease_LR', type = int, default = 20)

        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)
        args = parser.parse_args()

        model_dir = "checkpoints/" + args.dataset + '/'
        model_name = "ckpt.pth"
        train(DATASET = args.dataset, NEpochs = args.epochs,
        BatchSize = args.batch_size, learningRate = args.learning_rate, NDecreaseLR = args.decrease_LR, eps = args.eps,
        AMSGrad = args.AMSGrad, model_dir = model_dir, model_name = model_name, gpu = args.gpu)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
