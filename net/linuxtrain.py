from __future__ import print_function, division
import argparse
from datetime import datetime
from tool.dataset1 import mydataset
import json
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision import models
from tool import io
import torch
from net.test import create_model
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tool.logger import ConfusionMatrixBinary
from tool.loss import DiceLoss
PARSER = argparse.ArgumentParser(
    description='Pytorch Implementation of Network')
PARSER.add_argument('--modeldir', type=str, default='../model/',
                    help='directory to store the models (default: model/)')
PARSER.add_argument('--logdir', type=str, default='log',
                    help='directory to store tensorboard logs (default: log/)')
# Training config parameters
PARSER.add_argument('--epoch', type=int, default=100,
                    help='number of epochs to train (default: 100)')
PARSER.add_argument('--optimizer', type=str, default='sgd',
                    choices=['sgd', 'adagrad', 'adam'],
                    help='optimizer to use [sgd, adagrad, adam](default: sgd)')
PARSER.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.001)')
PARSER.add_argument('--momentum', type=float, default=0.99,
                    help='momentum (default: 0.99)')
PARSER.add_argument('--gamma', type=float, default=1.0,
                    help='learning rate decay (default: 1.0 (no decay))')
PARSER.add_argument('--step-size', type=int, default=7,
                    help='period of learning rate decay (default: 7)')

PARSER.add_argument('--device', type=str, default='cuda:0',
                    choices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu'],
                    help='device where to train the model, in GPU or in CPU \
                    (default: cuda:0)')
ARGS = PARSER.parse_args()
def train(model, criterion, optimizer, scheduler, dataloader_train,device):
    """Train and update parameters in the model.
    Returns:
        dict: Represents the statistics of the training. It will be
        in the following format:
        {'loss': loss, 'precision': precision,'mcc':mcc}
    """
    running_loss = 0
    dataset_size = 0

    scheduler.step() # step once in scheduler for learning rate decay

    for batched_sample in tqdm(dataloader_train):
        # convert the sample into proper data formats and send to GPU
        input = batched_sample['input'].float().to(device)
        labels1 = batched_sample['label'].cpu().data.numpy()
        labels = batched_sample['label'].long().to(device)
        #Convert not interest to no suck
        #print(labels.shape)  #[1,1,2,413]
        # zero the parameter gradients (required py PyTorch)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs, softmax_outputs = model(input)

            stat = ConfusionMatrixBinary()
            softmax_outputs = softmax_outputs[:,1,:,:]
            softmax_outputs = softmax_outputs.cpu().data.numpy()
            #print(softmax_outputs.shape)
            softmax_outputs[softmax_outputs>=0.5] = 1
            softmax_outputs[softmax_outputs < 0.5] = 0
            flatten_bool_label = (labels1== 1).flatten().astype(np.bool)
            flatten_bool_softmax = softmax_outputs.flatten().astype(np.bool)
            stat.update(flatten_bool_label,flatten_bool_softmax)
            #print(outputs.shape)  #[1,2,413,1]
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
           # print(loss.item())

        running_loss += loss.item() * input.size(0)
        dataset_size += len(batched_sample)
        # print(running_loss)
        # print(dataset_size)

    epoch_loss = running_loss / dataset_size
    epoch_pre = stat.get_precision()
    epoch_mcc = stat.get_mcc()

    return {'loss': epoch_loss,'pre':epoch_pre,'mcc':epoch_mcc}



def validate(model, criterion,dataloader_valid, device):
    """validate the model.
    Returns:
        dict: Statistics of the training. It will be in the following format:
        {'loss': loss, 'precision': precision}
    """
    running_loss = 0
    dataset_size = 0
    for batched_sample in tqdm(dataloader_valid):
        # convert the sample into proper data formats and send to GPU
        input = batched_sample['input'].float().to(device)
        # print(input.shape)
        labels1 = batched_sample['label'].cpu().data.numpy()
        labels = batched_sample['label'].long().to(device)
        with torch.set_grad_enabled(False):
            outputs, softmax_outputs = model(input)
            # print(outputs.shape)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * input.size(0)
        stat = ConfusionMatrixBinary()
        softmax_outputs = softmax_outputs[:, 1, :, :]
        softmax_outputs = softmax_outputs.cpu().data.numpy()
        softmax_outputs[softmax_outputs >= 0.5] = 1
        softmax_outputs[softmax_outputs < 0.5] = 0
        flatten_bool_label = (labels1 == 1).flatten().astype(np.bool)
        flatten_bool_softmax = softmax_outputs.flatten().astype(np.bool)
        stat.update(flatten_bool_label, flatten_bool_softmax)
        dataset_size += len(batched_sample)
    epoch_loss = running_loss / dataset_size
    epoch_pre = stat.get_precision()
    epoch_mcc = stat.get_mcc()

    return {'loss': epoch_loss,'pre':epoch_pre,'mcc':epoch_mcc}

def save_model(model, optimizer, epoch, val_precision, val_loss, modeldir,
               config_dict):

    io.save_checkpoint(
        {'epoch': epoch,
         'state_dict': model.state_dict(),
         'precision': val_precision,
         'loss': val_loss,
         'optimizer': optimizer.state_dict(),
         'config': config_dict},
        folder=modeldir,
        filename='network{}_{}.pth.tar'.format(epoch, val_precision))
def main(args):
    """Train the  model using the given user arguments.
    """
    config_dict = vars(args)
    model_dir = config_dict['modeldir']
    log_dir = config_dict['logdir']
    learning_rate = config_dict['lr']
    momentum = config_dict['momentum']
    gamma = config_dict['gamma']
    epochs = config_dict['epoch']
    device_name = config_dict['device']
    step_size = config_dict['step_size']
    optimizer_name = config_dict['optimizer']
    # build a Network
    model = create_model()
    # get train and test dataloaders
    train_data = mydataset('../CA', '../tool/train1.txt')
    dataloader_train = DataLoader(train_data, batch_size=1)

    valid_data = mydataset('../CA', '../tool/test.txt')
    dataloader_valid = DataLoader(valid_data, batch_size=1)

    # define the optimizer and the model parameters that will be optimized
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(model_params, lr=learning_rate, momentum=momentum)
    # scheduler = None
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # define the loss function (weight of class 1 (undecided) is 0.1)
    weight = torch.ones(2)
    weight[1]=0.1
    criterion = nn.CrossEntropyLoss(weight=weight)

    # send model and criterion to GPU
    device = torch.device(device_name)
    criterion.to(device)
    model.to(device)

    # train model
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        # fix batch_norm in the model to use image-net running statistics
        model.train()
        train_stats = train(model,criterion,optimizer,scheduler,dataloader_train,device)
        model.eval()
        test_stats = validate(model,criterion,dataloader_valid,device)
        save_model(model,optimizer,epoch,test_stats['pre'],test_stats['loss'],model_dir,config_dict)
        print('Train Loss: {:.4f}  Train Pre: {:.4f} Train MCC: {:.4f}'.format(
            train_stats['loss'],train_stats['pre'],train_stats['mcc']))
        print('Valid Loss: {:.4f}  Valid Pre: {:.4f} Valid MCC: {:.4f}'.format(
            test_stats['loss'],test_stats['pre'],test_stats['mcc']))

if __name__ == "__main__":
    main(ARGS)