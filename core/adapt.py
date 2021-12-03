import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from core import test, pretrain1
import params
from utils import make_cuda, mixup_data, save_model
from tqdm import tqdm


def train_tgt(source_cnn, target_cnn, critic,
              src_data_loader, tgt_data_loader, tgt_data_loader_eval):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    source_cnn.eval()
    target_cnn.encoder.train()
    critic.train()
    isbest = 0
    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    #target encoder
    optimizer_tgt = optim.Adam(target_cnn.parameters(),
                               lr=params.adp_c_learning_rate,
                               betas=(params.beta1, params.beta2),
                               weight_decay=params.weight_decay
                               )
    #Discriminator
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2),
                               weight_decay=params.weight_decay

                                  
                                  )

    ####################
    # 2. train network #
    ####################
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))
    val_acc = np.inf
    for epoch in range(params.num_epochs):

        print(f"Start Train Epoch {epoch + 1}")
        tqdm_dataset = tqdm(enumerate(zip(src_data_loader, tgt_data_loader)))

        general_loss_value = 0
        discrimiator_loss_value = 0
        matches = 0

        for step, ((images_src, _), (images_tgt, _)) in tqdm_dataset:

            # make images variable
            images_src = make_cuda(images_src)
            images_tgt = make_cuda(images_tgt)

            ###########################
            # 2.1 train discriminator #
            ###########################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = source_cnn.encoder(images_src)
            feat_tgt = target_cnn.encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = make_cuda(torch.zeros(feat_src.size(0)).long())
            label_tgt = make_cuda(torch.ones(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            general_loss_value += loss_critic.item()
            general_loss_value /= (step + 1)


            # optimize critic
            optimizer_critic.step()

            # 여기서 정확도 재는거 domain 정확도 아닌가? acc가 0.5로 나와야되고
            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            # matches += (pred_cls == label_concat).float().mean()
            matches += (pred_cls == label_concat).sum().item()


            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = target_cnn.encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_cuda(torch.zeros(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            # preds = torch.argmas= labels).sum().item()
            # matches /= (step + 1)

            discrimiator_loss_value += loss_tgt.item()
            discrimiator_loss_value /= (step + 1)



            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Iter' : step + 1,
                'General_Loss' : general_loss_value,
                'Discriminator_Loss' : discrimiator_loss_value,
                'ACC'  : matches / params.batch_size / 2 / (step + 1)
            })


        print("Start Validation")
        val_loss_value = pretrain1.eval(target_cnn, 0)

        if val_acc > val_loss_value:
            val_acc = val_loss_value            
            save_model(target_cnn, f"weights/{params.src_dataset}_adapt_{params.src_dataset}2{params.tgt_dataset}_{params.exp_name}.pt")
        print()

    # torch.save(critic.state_dict(), os.path.join(
    #     params.model_root,
    #     "ADDA-critic-final.pt"))
    # torch.save(target_cnn.state_dict(), os.path.join(
    #     params.model_root,
    #     "ADDA-target_cnn-final.pt"))
    return target_cnn