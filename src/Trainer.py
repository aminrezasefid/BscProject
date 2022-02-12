import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import pandas as pd
from torch_geometric.data import Data

from utils import  update_edge_time, calculate_edge_weight, update_edge_index,prepare_features

log_base = 1.5
val_batches = 72
target_dim = 3


def continuous_evaluation(data: Data, model,f_mode,n_f, epochs=100, lr=0.001, lr_discount=0.2, batch_size=9, mode="val"):
    """
    A gateway function for starting the training, validation and testing of the provided model using continuous
    evaluation
    :param data: a Dataset instance associated with the model that contains the necessary data for training and the
        metadata
    :param model: the model
    :param epochs: the number of epochs
    :param lr: the learning rate
    :param lr_discount: the learning rate discount if the adaptable learning rate is used
    :param batch_size: size of batches in which the model is trained
    :param mode: mode in which the function is used: "val" for validation, "test" for testing
    :return:
    """
    print("Continuous evaluation")
    train_function = train_cont
    test_function = test_cont

    if mode == "val":
        matches = data.matches.append(data.data_val, ignore_index=True)
    else:
        matches = data.data_test
        test_acc = []
        global val_batches
        val_batches = 1
    
    print(matches.shape[0])
    if f_mode=='fixed':
        prepare_features(data,matches,n_f)
    for i in range(0, matches.shape[0], batch_size):
        train_start_point = max(0, i - 40 * batch_size)
        if f_mode=='variable':
            prepare_features(data,matches[:i],n_f)
        test_function(data, model, matches.iloc[i:i + val_batches * batch_size])
        #print("{}: val: {},{} ----- train: {},{}".format(int(i / batch_size),i,i + val_batches * batch_size,train_start_point,i + batch_size))
        data.curr_time = train_start_point
        train_function(data,
                       model,
                       # matches.head(i + batch_size),
                       matches.iloc[train_start_point:i + batch_size],
                       #matches,
                       # epochs,
                       epochs + int(math.log(i + 1, log_base)),
                       # lr * (1 - lr_discount) ** int(i / batch_size / 50),
                       lr,
                       batch_size)
        
        print("T:{}, train_loss:{:.5f}, train_acc:{:.5f}, val_loss={:.5f}, val_acc={:.5f}"
              .format(int(i / batch_size),
                      data.train_loss[-1],
                      data.train_accuracy[-1],
                      data.val_loss[-1],
                      data.val_accuracy[-1]))
        if mode == "test":
            test_acc.append(data.val_accuracy[-1])

    val_acc = data.val_accuracy[len(data.val_accuracy) - val_batches:]
    data.val_acc = sum(val_acc) / len(val_acc)

    if mode == "test":
        print(sum(test_acc) / len(test_acc))
    else:
        print(data.val_acc)


def train_cont(data: Data, model: torch.nn.Module, matches,
               epochs:int = 100, lr: int = 0.0001, batch_size:int = 9, print_info: bool = False):
    """
    A function for training the provided model with the provided matches using continuous evaluation
    :param data: a Dataset instance associated with the model that contains the necessary data for training and the
        metadata
    :param model: the model
    :param matches: the data set
    :param epochs: the number of epochs
    :param lr: the learning rate
    :param batch_size: size of batches in which the model is trained
    :param print_info: a binary flag that indicates if the information about the training should be printed out to the
        terminal
    :return:
    """
    criterion = nn.NLLLoss()  # weight=torch.tensor([1.6,1.95,1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = []
    running_accuracy = []
    home_win = 0
    for epoch in range(epochs):
        acc = 0
        loss_value = 0.0
        optimizer.zero_grad()
        for j in range(0, matches.shape[0], batch_size):
            home, away, result = torch.from_numpy(matches.iloc[j:j + batch_size]['HomeTeam'].values.astype('int64')), \
                                 torch.from_numpy(matches.iloc[j:j + batch_size]['AwayTeam'].values.astype('int64')), \
                                 torch.from_numpy(
                                     matches.iloc[j:j + batch_size]['lwd'].values.astype('int64').reshape(-1, ))
            home_win += (result == 2).sum().item()
            # label = torch.zeros(result.shape[0], target_dim).scatter_(1, torch.tensor(result), 1)  # one-hot label for loss
            outputs = model(data, home, away)
            #print("epoch:",epoch,"-----edgie size:",data.edge_index.shape)
            # loss = criterion(outputs, label.to(torch.float))
            loss = criterion(outputs, result)
            loss.backward()
            optimizer.step()
            loss_value += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = int((predicted == result).sum().item())
            running_accuracy.append(correct)
            acc += correct

            update_edge_time(data, home, away, result)
            update_edge_index(data)
            calculate_edge_weight(data)
            data.curr_time += 1

        if print_info:
            print("Epoch:{}, train_loss:{:.5f}, train_acc:{:.5f}"
                  .format(epoch, loss_value, acc / (matches.shape[0])))

        data.curr_time -= math.ceil(matches.shape[0] / batch_size)  # probably is safe to be set to 0 each epoch
        running_loss.append(loss_value)
        # if epoch % 50 == 49:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.8
    # print(home_win/(matches.shape[0] * epochs))
    data.train_loss.append(sum(running_loss) / ((matches.shape[0] / batch_size) * epochs))
    data.train_accuracy.append(sum(running_accuracy) / (matches.shape[0] * epochs))


def test_cont(data: Data, model: torch.nn.Module, matches, mode: str = "val"):
    """
    A function for testing the provided model on the provided matches using continuous evaluation
    :param data: a Dataset instance associated with the model that contains the necessary data for training and the
        metadata
    :param model: the model
    :param matches: the data set
    :param mode: mode in which the testing function is used: "val" for validation, "test" for testing
    :return:
    """
    criterion = nn.NLLLoss()  # weight=torch.tensor([1.6,1.95,1])

    predicted, label, outputs = get_predictions(data, model, matches)
    drawCount=predicted[predicted==1].shape[0]
    if drawCount!=0:
        print(drawCount)
    loss = criterion(outputs, label).item()

    correct = int((predicted == label).sum().item())
    if mode == "test":
        data.test_accuracy = float(correct) / matches.shape[0]
    else:
        data.val_accuracy.append(float(correct) / matches.shape[0])
        data.val_loss.append(loss)


def get_predictions(data: Data, model: torch.nn.Module, matches) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the predictions for the provided matches using provided model
    :param data: a Dataset instance associated with the model that contains the necessary data for training and the
        metadata
    :param model: the model
    :param matches: the data set
    :return: Tensors of predictions for each match, a ground truth label and the probabilities
    """
    outputs, label = get_probabilities(data, model, matches)
    _, predicted = torch.max(torch.exp(outputs.data), 1)
    return predicted, label, outputs


def get_probabilities(data, model, matches):
    model.eval()
    home, away, label = torch.from_numpy(matches['HomeTeam'].values.astype('int64')), \
                        torch.from_numpy(matches['AwayTeam'].values.astype('int64')), \
                        torch.from_numpy(matches['lwd'].values.astype('int64').reshape(-1, ))
    with torch.no_grad():
        outputs = model(data, home, away)
    model.train()
    return outputs, label
