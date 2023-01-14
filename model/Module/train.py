
"""
실행 명령어 : python train.py --batch-size 16 --wandb True --model baseline --normalize none --tag baseline --seed 1004 --descript "baseline resnet18 speaker independent"

wav loader : python train.py --batch-size 16 --wandb True --model wav_res --tag baseline --seed 1004 --descript "baseline resnet18 speaker independent"

"""

import wandb
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn # 인공 신경망 모델들 모아놓은 모듈
import torch.nn.functional as F #그중 자주 쓰이는것들을 F로
import cv2
import pandas as pd
import os
from glob import glob
import torchvision.models as models
import sys
import argparse

from Utils.pytorchtools import EarlyStopping # 상위 폴더에 추가된 모듈.
from Dataset.cv_spliter import cv_spliter #데이터 분할 모듈
from Dataset.Data import make_data # phrase 데이터를 담아둔다.
from Dataset.Dataset import load_data,load_test_data
from Model.Models import model_initialize
from Utils.Utils import get_mean_std,save_result



import torchaudio
import json
#import torchaudio.functional as F





# # SVD 문장 데이터에서 Feature 추출
# - mfcc


import numpy as np
import librosa, librosa.display 
import matplotlib.pyplot as plt
#window sizde : FFT를 할때 참조할 그래프 길이 ( 프레임 하나당 sample 수 )
#자연어 처리에서는 25ms 사용. https://ahnjg.tistory.com/93
#초당 50000hz 중 1250개씩 윈도우 사이즈로 사용.

#default param

# spectro_run_config = dict(
#     sr=16000,
#     n_fft=350,
#     hop_length=50,
#     win_length=350,
#     # training
#     batch_size=16,
# )

# #기존 파라미터
# #mel_run_config = dict(
# #    sr=16000,
# #    n_mels=128,
# #    win_length =  300,
# #    n_fft= 2048,
# #    hop_length= 50,
# #    f_max = 8000    
# #)
# mel_run_config = dict(
#    sr=16000,
#    n_mels=128,
#    win_length =  300,
#    n_fft= 2048,
#    hop_length= 50,
#    f_max = 8000    
# )


# mfcc_run_config = dict(
#     sr=16000,
#     n_mfcc=27,
#     #dct_type=3, # type2 default
#     lifter = 35,
    
#     #mel spectro
#     n_mels=170,
#     hop_length=750,
#     n_fft =14056,    
#     win_length=1100,
#     f_max=8000,
    
#     # training
#     #batch_size=32,
#     mel_scale ='htk',
    
#     # data
# )

#confusion matrix 계산
#test set 계산.
def test_evaluate(model,model_name,test_loader,DEVICE,criterion,save_result=False):
    model.eval()
    test_loss = 0
    predictions = []
    answers = []

    #save result용
    output_list = []
    file_list = []

    #no_grad : 그래디언트 값 계산 막기.
    if model_name == 'baseline':
        with torch.no_grad():
            for image,label in test_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                test_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                answers +=label
                predictions +=prediction
    elif model_name == 'wav_res':
        with torch.no_grad():
            for image,label,path_list in test_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                test_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                answers +=label
                predictions +=prediction

                #save result
                softmax_outputs = F.softmax(output,dim=1)[:,1] # pathology 확률 
                output_list+= softmax_outputs
                file_list += path_list

    elif model_name == 'wav_res_latefusion':
        with torch.no_grad():
            for image,label,path_list in test_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                test_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                answers +=label
                predictions +=prediction

                #save result
                softmax_outputs = F.softmax(output,dim=1)[:,1] # pathology 확률 
                output_list+= softmax_outputs
                file_list += path_list

    elif model_name == 'wav_res_concat':
        with torch.no_grad():
            for image,label,path_list in test_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                test_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                answers +=label
                predictions +=prediction

                #save result
                softmax_outputs = F.softmax(output,dim=1)[:,1] # pathology 확률 
                output_list+= softmax_outputs
                file_list += path_list
    elif model_name == 'wav_res_concat_latefusion':
        with torch.no_grad():
            for image,label,path_list in test_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                test_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                answers +=label
                predictions +=prediction

                #save result
                softmax_outputs = F.softmax(output,dim=1)[:,1] # pathology 확률 
                output_list+= softmax_outputs
                file_list += path_list
    elif model_name == 'wav_res_concat_allfusion':
        with torch.no_grad():
            for image,label,path_list in test_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                test_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                answers +=label
                predictions +=prediction

                #save result
                softmax_outputs = F.softmax(output,dim=1)[:,1] # pathology 확률 
                output_list+= softmax_outputs
                file_list += path_list
                
    elif model_name == 'msf':
        with torch.no_grad():
            for image,mfccs,label in test_loader:
                image = image.to(DEVICE)
                mfccs = mfccs.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image,mfccs)
                test_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                answers +=label
                predictions +=prediction
    elif model_name == 'decomp':
        with torch.no_grad():
            for image,label in test_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                test_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                answers +=label
                predictions +=prediction
    else:
        with torch.no_grad():
            for image,label in test_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                test_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                answers +=label
                predictions +=prediction
    
    if save_result:
        print("save result")
        return predictions,answers,test_loss,output_list,file_list

    return predictions,answers,test_loss


#8. 학습
def train(model,model_name,train_loader,optimizer,DEVICE,criterion):
    model.train()
    correct = 0
    train_loss = 0
    if model_name == 'baseline':
        for batch_idx,(image,label) in enumerate(train_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    elif model_name == 'wav_res':
        for batch_idx,(image,label,path_list) in enumerate(train_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    elif model_name == 'wav_res_latefusion':
        for batch_idx,(image,label,path_list) in enumerate(train_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    elif model_name == 'wav_res_concat':
        for batch_idx,(image,label,path_list) in enumerate(train_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    elif model_name == 'wav_res_concat_latefusion':
        for batch_idx,(image,label,path_list) in enumerate(train_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    elif model_name == 'wav_res_concat_allfusion':
        for batch_idx,(image,label,path_list) in enumerate(train_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    elif model_name =='msf':
        for batch_idx,(image,mfccs,label) in enumerate(train_loader):
            image = image.to(DEVICE)
            mfccs = mfccs.to(DEVICE)
            label = label.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image,mfccs) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    elif model_name == 'decomp':
        for batch_idx,(image,label) in enumerate(train_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    else:
        for batch_idx,(image,label) in enumerate(train_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.        
    train_loss/=len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    return train_loss,train_accuracy



#9. 학습 진행하며, validation 데이터로 모델 성능확인
def evaluate(model,model_name,valid_loader,DEVICE,criterion):
    model.eval()
    valid_loss = 0
    correct = 0
    #no_grad : 그래디언트 값 계산 막기.
    if model_name == 'baseline':
        with torch.no_grad():
            for image,label in valid_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #true.false값을 sum해줌. item
    elif model_name == 'wav_res':
        with torch.no_grad():
            for image,label,path_list in valid_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                #print(F.softmax(output))
                #print(output.size())
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #print(prediction.eq(label.view_as(prediction)))
                #print(path_list)
                #true.false값을 sum해줌. item
    elif model_name == 'wav_res_latefusion':
        with torch.no_grad():
            for image,label,path_list in valid_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #true.false값을 sum해줌. item
    elif model_name == 'wav_res_concat':
        with torch.no_grad():
            for image,label,path_list in valid_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #true.false값을 sum해줌. item
    elif model_name == 'wav_res_concat_latefusion':
        with torch.no_grad():
            for image,label,path_list in valid_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #true.false값을 sum해줌. item
    elif model_name == 'wav_res_concat_allfusion':
        with torch.no_grad():
            for image,label,path_list in valid_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #true.false값을 sum해줌. item
    elif model_name =='msf':
        with torch.no_grad():
            for image,mfccs,label in valid_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                mfccs = mfccs.to(DEVICE)
                output = model(image,mfccs)
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #true.false값을 sum해줌. item
    elif model_name == 'decomp':
        with torch.no_grad():
            for image,label in valid_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #true.false값을 sum해줌. item
    else:
        with torch.no_grad():
            for image,label in valid_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #true.false값을 sum해줌. item        
    valid_loss /= len(valid_loader.dataset)
    valid_accuracy = 100. * correct / len(valid_loader.dataset)
    return valid_loss,valid_accuracy






def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='Voice Disorder Detection Trainer')
    parser.add_argument('--batch-size', type=int, default=32, metavar='batch',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=40, metavar='EPOCH',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')                  
    parser.add_argument('--wandb',type=bool, default=False,
                        help='Use wandb log')                        
    parser.add_argument('--model',type=str, default='baseline',
                        help='list : [msf, baseline,wav_res,wav_res_latefusion,wav_res_concat,wav_res_concat_latefusion,wav_res_concat_allfusion]')
    parser.add_argument('--dataset',type=str, default='phrase',
                        help='list : [phrase, a_h, a_n, a_l, a_fusion ... ]')
    parser.add_argument('--name',type=str, default='res18',
                            help='write custom model for wandb')
    parser.add_argument('--normalize',type=bool,default=False,
                        help='true or false to get std normalize')
    parser.add_argument('--project-name',type=str, default='SVD-voice-disorder',
                            help='project name for wandb')
    parser.add_argument("--augment", nargs='+', type=str,help="[crop,spec_augment]",default=[])
    parser.add_argument('--tag',type=str,default='',nargs='+',help='tag for wandb')
    parser.add_argument('--seed',type=int,default=1004,help='set the test seed')
    parser.add_argument('--save-result',type=bool,default=False,help='save result of validation set')
    parser.add_argument('--descript',type=str, default='baseline. speaker indep',
                            help='write config for wandb')

    args = parser.parse_args()


    if args.wandb:
        project_name = args.project_name
        wandb.init(project=project_name, entity="bub3690",tags=args.tag,settings=wandb.Settings(_disable_stats=True))
        wandb_run_name = args.model+'_'+args.name+'_norm_'+str(args.normalize).lower()+'_seed_'+str(args.seed)
        wandb.run.name = wandb_run_name
        wandb.run.save()
        wandb.run.summary.update({"seed" : args.seed,})


    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    #DEVICE = torch.device('cpu')
    print('Using Pytorch version : ',torch.__version__,' Device : ',DEVICE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1234)

    random_state = args.seed # 1004,1005,1006,1007,1008
    

    
    speaker_file_path = "../../voice_data/only_organics_healthy_available_ver2.xlsx" # 퓨전셋에 맞게 01.10 수정
    speaker_file_path_abs = os.path.abspath(speaker_file_path)

    X_train_list, X_valid_list, X_test, Y_train_list, Y_valid_list, Y_test = cv_spliter(random_state,speaker_file_path_abs)
        


    # # 데이터로더


    #3. 하이퍼 파라미터
    BATCH_SIZE =  args.batch_size #한 배치당 32개 음성데이터
    EPOCHS = 40 # 전체 데이터 셋을 40번 반복
    lr=1e-4
    augment_kind=args.augment
    weight_decay = 0

    augment_params = dict()
    augment_params['spec_augment']=[30, 13, 1.0] # time_mask_param,freq_mask_param, augement_p
    augment_params['crop']=[1, 50, 1.0] # augment_n , augment_size, augement_p


    if args.wandb:
        wandb.config.update({
            "learning_rate": lr,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "augment":augment_kind,
            "weight_decay":weight_decay,
            "특이사항":args.descript,
        })

    data_instance=make_data(args.dataset) # pickle file 로드
    

    ## hyp-parameter
    print("하이퍼 파라미터 로드")
    with open('./Params/log-spectrogram.json') as f:
        spectro_run_config = json.load(f)[str(args.seed)] 
    with open('./Params/mel-spectrogram.json') as f:
        mel_run_config = json.load(f)[str(args.seed)] 
    with open('./Params/mfcc.json') as f:
        mfcc_run_config = json.load(f)[str(args.seed)] 


    ## find mean, std
    ## log_spectro, mel_spectro, mfcc 순으로 담긴다.
    norm_mean_list = []
    norm_std_list = []
    
    if args.normalize:
        print("normalize 시작")
        spectro_mean,spectro_std = get_mean_std(X_train_list[0]+X_valid_list[0], Y_train_list[0]+Y_valid_list[0],'logspectrogram',spectro_run_config,mel_run_config,mfcc_run_config)        
        mel_mean,mel_std = get_mean_std(X_train_list[0]+X_valid_list[0], Y_train_list[0]+Y_valid_list[0], 'melspectrogram',spectro_run_config,mel_run_config,mfcc_run_config)
        mfcc_mean,mfcc_std = get_mean_std(X_train_list[0]+X_valid_list[0], Y_train_list[0]+Y_valid_list[0],'mfcc',spectro_run_config,mel_run_config,mfcc_run_config)

        norm_mean_list = [spectro_mean,mel_mean,mfcc_mean]
        norm_std_list = [spectro_std,mel_std,mfcc_std]
        print("mean : ",norm_mean_list)
        print("std : ",norm_std_list)
    

    ##### 10. 학습 및 평가.
    # resnet18 pretrained true
    # kfold 적용

    train_accs = []
    valid_accs = []


    for data_ind in range(1,6):

        check_path = './checkpoint/checkpoint_ros_fold_'+str(data_ind)+'_'+args.model+'_seed_'+str(args.seed)+'_dataset_'+args.dataset+'_norm_'+str(args.normalize).lower()+'_organics_speaker.pt'
        print(check_path)
        early_stopping = EarlyStopping(patience = 5, verbose = True, path=check_path)
        train_loader,validation_loader = load_data( X_train_list[data_ind-1],
                                                    X_valid_list[data_ind-1],
                                                    Y_train_list[data_ind-1],
                                                    Y_valid_list[data_ind-1],
                                                    BATCH_SIZE,
                                                    spectro_run_config,
                                                    mel_run_config,
                                                    mfcc_run_config,
                                                    args.normalize,
                                                    norm_mean_list,
                                                    norm_std_list,
                                                    args.model,
                                                    args.dataset,
                                                    args.augment,
                                                    augment_params)
        best_train_acc=0 # accuracy 기록용
        best_valid_acc=0
        
        model = model_initialize(args.model,  spectro_run_config,mel_run_config,mfcc_run_config)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
        
        print("[{} 교차검증] 학습 시작\n ----- ".format(data_ind))
        for Epoch in range(1,EPOCHS+1):
            train_loss,train_accuracy=train(model,args.model,train_loader,optimizer,DEVICE,criterion)
            valid_loss,valid_accuracy = evaluate(model,args.model,validation_loader,DEVICE,criterion)
            
            logger_valid_acc = "valid {}fold Accuracy".format(data_ind)
            logger_train_acc = "train {}fold Accuracy".format(data_ind)
            logger_valid_loss = "valid {}fold loss".format(data_ind)
            logger_train_loss = "train {}fold loss".format(data_ind)
            

            if args.wandb:
                wandb.log({
                        #logger_train_acc :train_accuracy,
                        #logger_train_loss : train_loss,
                        logger_valid_acc : valid_accuracy,
                        logger_valid_loss : valid_loss},
                        commit=False,
                        step=Epoch)

            print("\n[EPOCH:{}]\t Train Loss:{:.4f}\t Train Acc:{:.2f} %  | \tValid Loss:{:.4f} \tValid Acc: {:.2f} %\n".
                format(Epoch,train_loss,train_accuracy,valid_loss,valid_accuracy))
            

            early_stopping(valid_loss, model)
            if -early_stopping.best_score == valid_loss:
                best_train_acc, best_valid_acc = train_accuracy,valid_accuracy
                if args.wandb:
                    wandb.run.summary.update({"best_valid_{}fold_acc".format(data_ind) : best_valid_acc})
            
            if early_stopping.early_stop:
                    train_accs.append(best_train_acc)
                    valid_accs.append(best_valid_acc)
                    print("[{} 교차검증] Early stopping".format(data_ind))
                    break

            if Epoch==EPOCHS:
                #만약 early stop 없이 40 epoch라서 중지 된 경우. 
                train_accs.append(best_train_acc)
                valid_accs.append(best_valid_acc)
            #scheduler.step()
            #print(scheduler.get_last_lr())
        # if args.save_result:
        #     print("save valid result")
        #     model=model_initialize(args.model,  spectro_run_config,mel_run_config,mfcc_run_config)
        #     check_path = './checkpoint/checkpoint_ros_fold_'+str(data_ind)+'_'+args.model+'_seed_'+str(args.seed)+'_dataset_'+args.dataset+'_norm_'+str(args.normalize).lower()+'_organics_speaker.pt'
        #     model.load_state_dict(torch.load(check_path))            
        #     test_loader = load_test_data(
        #                                     X_valid_list[data_ind-1],
        #                                     Y_valid_list[data_ind-1],
        #                                     BATCH_SIZE,
        #                                     spectro_run_config,
        #                                     mel_run_config,
        #                                     mfcc_run_config,    
        #                                     args.normalize,
        #                                     norm_mean_list,
        #                                     norm_std_list,
        #                                     args.model,
        #                                     args.dataset
        #                                 )
        #     predictions,answers,test_loss,validation_outputs,validation_files = test_evaluate(model,args.model,test_loader, DEVICE, criterion,save_result=True)
            
        #     print(validation_files)
        #     print(predictions)
        #     print(answers)
        #     print(validation_outputs)        

    # # Model 결과 확인
    sum_valid=0
    for data_ind in range(5):
        print("[{} 교차검증] train ACC : {:.4f} |\t valid ACC: {:.4f} ".format(data_ind+1,train_accs[data_ind],valid_accs[data_ind] ))
        sum_valid+=valid_accs[data_ind]
        
    print("평균 검증 정확도",sum_valid/5,"%")
    

    # validation result

    # for save result
    all_filename = []
    all_prediction= []
    all_answers= []
    all_probs = []

    if args.save_result:
        print("save valid result")
        for data_ind in range(1,6):
            model=model_initialize(args.model,  spectro_run_config,mel_run_config,mfcc_run_config)
            check_path = './checkpoint/checkpoint_ros_fold_'+str(data_ind)+'_'+args.model+'_seed_'+str(args.seed)+'_dataset_'+args.dataset+'_norm_'+str(args.normalize).lower()+'_organics_speaker.pt'
            model.load_state_dict(torch.load(check_path))            
            test_loader = load_test_data(
                                            X_valid_list[data_ind-1],
                                            Y_valid_list[data_ind-1],
                                            BATCH_SIZE,
                                            spectro_run_config,
                                            mel_run_config,
                                            mfcc_run_config,    
                                            args.normalize,
                                            norm_mean_list,
                                            norm_std_list,
                                            args.model,
                                            args.dataset
                                        )
            predictions,answers,test_loss,validation_outputs,validation_files = test_evaluate(model,args.model,test_loader, DEVICE, criterion,save_result=True)

            all_filename.append(validation_files)
            all_prediction.append(predictions)
            all_answers.append(answers)
            all_probs.append(validation_outputs)
            #import pdb;pdb.set_trace()
        save_result(all_filename,all_prediction, all_answers,all_probs,speaker_file_path_abs,args)







    # # Model Test
    # 
    # - test set
    # - confusion matrix

                
    # Confusion matrix (resnet18)
    # kfold의 confusion matrix는 계산 방법이 다르다.
    # 모델을 각각 불러와서 test set을 평가한다.

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score


    test_loader = load_test_data(
        X_test,
        Y_test,
        BATCH_SIZE,
        spectro_run_config,
        mel_run_config,
        mfcc_run_config,    
        args.normalize,
        norm_mean_list,
        norm_std_list,
        args.model,
        args.dataset)


    cf = np.zeros((2,2))
    cf_list = []
    average_accuracy = 0
    average_fscore = 0
    average_uar = 0

    for data_ind in range(1,6):
        model=model_initialize(args.model,  spectro_run_config,mel_run_config,mfcc_run_config)
        check_path = './checkpoint/checkpoint_ros_fold_'+str(data_ind)+'_'+args.model+'_seed_'+str(args.seed)+'_dataset_'+args.dataset+'_norm_'+str(args.normalize).lower()+'_organics_speaker.pt'
        model.load_state_dict(torch.load(check_path))

        predictions,answers,test_loss = test_evaluate(model,args.model,test_loader, DEVICE, criterion)
        predictions=[ dat.cpu().numpy() for dat in predictions]
        answers=[ dat.cpu().numpy() for dat in answers]

        
        cf = confusion_matrix(answers, predictions)
        cf_list.append(cf)
        
        acc = accuracy_score(answers,predictions)
        average_accuracy+=acc
        
        recall=cf[1,1]/(cf[1,1]+cf[1,0])
        specificity=cf[0,0]/(cf[0,0]+cf[0,1])

        average_uar += (specificity+recall)/2
        #fscore=2*precision*recall/(precision+recall)
        
        #fscroe macro추가
        fscore = f1_score(answers,predictions,average='macro')
        average_fscore+=fscore
        
        print('{}번 모델'.format(data_ind))
        print("Accuracy : {:.4f}% ".format(acc*100))
        #print("Precision (pathology 예측한 것중 맞는 것) : {:.4f}".format(precision))
        print("recall (실제 pathology 중  예측이 맞는 것) : {:.4f}".format(recall))
        print("specificity : {:.4f}%".format(specificity))
        print("UAR : {:.4f}%".format( (specificity+recall)/2 ))
        
        
        print("f score : {:.4f} ".format(fscore))
        print(cf)
        print("-----")

    print("평균 acc : {:.4f}".format(average_accuracy/5))
    print("평균 UAR : {:.4f}".format(average_uar/5))
    print("평균 f1score : {:.4f}".format(average_fscore/5))
    if args.wandb:
        wandb.run.summary.update({"test 평균 acc" : average_accuracy/5})
        wandb.run.summary.update({"test 평균 f1" : average_fscore/5})
        wandb.run.summary.update({"test 평균 UAR" : average_uar/5})
    return



if __name__=='__main__':
    main()