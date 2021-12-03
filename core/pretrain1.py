import torch.nn as nn
import torch.optim as optim
import torch
import params
import numpy as np
from utils import make_cuda, save_model, LabelSmoothingCrossEntropy,mixup_data
from random import *
import sys
from tqdm import tqdm
import time
from copy import copy
def train_src(model, source_data_loader, source_eval_data_loader=None, model_name=None):
    

    optimizer = optim.Adam(
        model.parameters(),
        lr=params.pre_c_learning_rate,
        betas=(params.beta1, params.beta2),
        weight_decay=params.weight_decay
        )

    if params.labelsmoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing= params.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    val_acc = np.inf
    for epoch in range(params.num_epochs_pre):
        
        print(f"Start Train Epoch {epoch + 1}")
        tqdm_dataset = tqdm(enumerate(source_data_loader))
        
        model.train()
        loss_value = 0
        matches = 0
        
        for step, (images, labels) in tqdm_dataset:
            # make images and labels variable
            images = make_cuda(images)
            labels = make_cuda(labels.squeeze_())
            # zero gradients for optimizer
            optimizer.zero_grad()
            
            # compute loss for critic
            outs = model(images)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            preds = torch.argmax(outs, dim=-1)
            matches += (preds == labels).sum().item()
            # matches /= (step + 1)

            loss_value += loss.item()
            loss_value /= (step + 1)
            # optimize source classifier

            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Iter' : step + 1,
                'Loss' : loss_value,
                'ACC'  : matches / params.batch_size / (step + 1)
            })

        if not source_eval_data_loader:
            continue


        print("Start Validation")
        val_loss_value = eval(model, source_eval_data_loader)
        if val_acc > val_loss_value:
            val_acc = val_loss_value
            if model_name:            
                save_model(model, model_name)
            else:
                best_model = deepcopy(model)
        print()

    return best_model


def eval(model, eval_dataset):

    tqdm_dataset = tqdm(enumerate(eval_dataset))

    if params.labelsmoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing= params.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    val_loss_value = 0
    val_matches = 0
    with torch.no_grad():
        model.eval()

        for step, (images, labels) in tqdm_dataset:
            images = make_cuda(images)
            labels = make_cuda(labels.squeeze_())

            outs = model(images)
            loss = criterion(outs, labels)

            preds = torch.argmax(outs, dim=-1)
            val_matches += (preds == labels).sum().item()

            val_loss_value += loss.item()
            val_loss_value /= (step + 1)

            tqdm_dataset.set_postfix({
            'Iter' : step + 1,
            'Loss' : val_loss_value,
            'ACC'  : val_matches / params.batch_size / (step + 1)
            })

    return val_loss_value