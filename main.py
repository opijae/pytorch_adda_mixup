import params
from utils import get_data_loader, init_model, init_random_seed,mixup_data, save_tensor_img_with_label
from core import pretrain , adapt , test,mixup, pretrain1
import torch
from models.models import *
import numpy as np
import sys
import cv2
import warnings
warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)
    print(f"Is cuda availabel? {torch.cuda.is_available()}")



    #set loader
    print("src data loader....")
    src_data_loader = get_data_loader(params.src_dataset,adp=False,size = 10000)
    src_data_loader_eval = get_data_loader(params.src_dataset,train=False)
    print("tgt data loader....")
    tgt_data_loader = get_data_loader(params.tgt_dataset,adp=False,size = 50000)
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)
    print(f"scr data size : {len(src_data_loader.dataset)}")
    print(f"tgt data size : {len(tgt_data_loader.dataset)}")

    # check image
    save_tensor_img_with_label(next(iter(src_data_loader)), 'src')
    save_tensor_img_with_label(next(iter(src_data_loader_eval)), 'src_eval')
    save_tensor_img_with_label(next(iter(tgt_data_loader)), 'tgt')
    save_tensor_img_with_label(next(iter(tgt_data_loader_eval)), 'tgt_eval')

    print("start training")
    source_cnn = CNN(in_channels=3).to("cuda")
    target_test_cnn = CNN(in_channels=3).to("cuda")  # train on target
    target_cnn = CNN(in_channels=3, target=True).to("cuda")
    discriminator = Discriminator().to("cuda")

    print("#######pre-train#######")

    model_save_path = f"weights/{params.exp_name}.pt"

    if params.pretrain:
        source_cnn = pretrain1.train_src(source_cnn, src_data_loader, src_data_loader_eval, model_name = model_save_path)
    source_cnn.load_state_dict(torch.load(model_save_path))

    print("#######pre-train_tgt_cnn#######")
    target_test_cnn = pretrain1.train_src(target_test_cnn, tgt_data_loader, tgt_data_loader_eval)


    print('######eval_test#####')
    print(f"Pretrained src model({params.src_dataset}) -> tgt dataset({params.tgt_dataset})")
    pretrain1.eval(source_cnn, tgt_data_loader_eval)
    pretrain1.eval(target_test_cnn, tgt_data_loader_eval)

    target_cnn.load_state_dict(source_cnn.state_dict())
    
    tgt_encoder = adapt.train_tgt(source_cnn, target_cnn, discriminator,
                            src_data_loader,tgt_data_loader,tgt_data_loader_eval)




    print("=== Evaluating classifier for encoded target domain ===")
    print(f"{params.src_dataset} -> {params.tgt_dataset} ")

    print("Eval | source_cnn | src_data_loader_eval")    
    pretrain1.eval(source_cnn, src_data_loader_eval)

    print(">>> Eval | source_cnn | tgt_data_loader_eval <<<")    
    pretrain1.eval(source_cnn, tgt_data_loader_eval)

    print(">>> Eval | target_cnn | tgt_data_loader_eval <<<")    
    pretrain1.eval(target_cnn, tgt_data_loader_eval)