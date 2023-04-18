import torch
import numpy as np
import torch.nn as nn # 인공 신경망 모델들 모아놓은 모듈
import torch.nn.functional as F #그중 자주 쓰이는것들을 F로
import pandas as pd
import torchvision.models as models
from tqdm import tqdm




def pad_samples(sig_tensor):
    length = 16000*6 #sample rate *2 padding을 위한 파라미터 (하이퍼 파라미터로인해 사이즈는 계속 바뀐다.)
    pad1d = lambda a, i: a[:,0:i] if a.shape[1] > i else torch.hstack((a, torch.zeros((1,i-a.shape[1]))))        
    sig_tensor = pad1d(sig_tensor,length)
    print(sig_tensor.size())
    return pad_samples

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def splicing_criterion(criterion, pred, y_a, y_b, lam):
    #print(lam * criterion(pred, y_a))
    #print(lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b))
    return (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)).mean()


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
            for image,label,path_list,origin_length in test_loader:
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
    elif model_name == 'wav_res_smile':
        with torch.no_grad():
            for image,handcrafted,label,path_list,origin_length in test_loader:
                image = image.to(DEVICE)
                handcrafted = handcrafted.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image,handcrafted)
                test_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                answers +=label
                predictions +=prediction

                #save result
                softmax_outputs = F.softmax(output,dim=1)[:,1] # pathology 확률 
                output_list+= softmax_outputs
                file_list += path_list
    elif model_name == 'wav_vgg16_smile':
        with torch.no_grad():
            for image,handcrafted,label,path_list,origin_length in test_loader:
                image = image.to(DEVICE)
                handcrafted = handcrafted.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image,handcrafted)
                test_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                answers +=label
                predictions +=prediction

                #save result
                softmax_outputs = F.softmax(output,dim=1)[:,1] # pathology 확률 
                output_list+= softmax_outputs
                file_list += path_list
    elif model_name == 'wav_mlp_smile':
        with torch.no_grad():
            for image,handcrafted,label,path_list,origin_length in test_loader:
                image = image.to(DEVICE)
                handcrafted = handcrafted.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image,handcrafted)
                test_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                answers +=label
                predictions +=prediction

                #save result
                softmax_outputs = F.softmax(output,dim=1)[:,1] # pathology 확률 
                output_list+= softmax_outputs
                file_list += path_list                
    elif model_name == 'wav_res_phrase_eggfusion_lstm':
        with torch.no_grad():
            for image,label,path_list,origin_length in test_loader:
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
    elif model_name == 'wav_bixception_phrase_eggfusion_lstm':
        with torch.no_grad():
            for image,label,path_list,origin_length in test_loader:
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
    elif model_name == 'wav_res_phrase_eggfusion_mmtm':
        with torch.no_grad():
            for image,label,path_list,origin_length in test_loader:
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
    elif model_name == 'wav_res_phrase_eggfusion_mmtm_bam':#wav_res_phrase_eggfusion_mmtm_nonlocal
        with torch.no_grad():
            for image,label,path_list,origin_length in test_loader:
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
    elif model_name == 'wav_res_phrase_eggfusion_mmtm_nonlocal':
        with torch.no_grad():
            for image,label,path_list,origin_length in test_loader:
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
    elif model_name == 'wav_res_time_attention':
        with torch.no_grad():
            for image,label,path_list,origin_length in test_loader:
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
    elif model_name == 'wav_res_splicing':
        with torch.no_grad():
            for image,label,path_list,origin_length in test_loader:
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
    elif model_name == 'wav_res_mixup':
        with torch.no_grad():
            for image,label,path_list,origin_length in test_loader:
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
    elif model_name == 'wav_res_concat_phrase_vowel':
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
    elif model_name == 'wav_res_concat_allfusion_attention':
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
    elif model_name == 'wav_res_latefusion_phrase_vowel':
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
            for image,label,path_list,origin_length in test_loader:
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
    if save_result:
        print("save result")
        return predictions,answers,test_loss,output_list,file_list

    return predictions,answers,test_loss


#8. 학습
def train(model,model_name,train_loader,optimizer,DEVICE,criterion):
    model.train()
    correct = 0
    train_loss = 0
    total = 0 # mixup에 이용
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
        for batch_idx,(image,label,path_list,origin_length) in enumerate(train_loader):
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
    elif model_name == 'wav_res_smile':
        for batch_idx,(image,handcrafted,label,path_list,origin_length) in tqdm(enumerate(train_loader)):
            image = image.to(DEVICE)
            handcrafted = handcrafted.to(DEVICE)
            label = label.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image,handcrafted) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    elif model_name == 'wav_vgg16_smile':
        for batch_idx,(image,handcrafted,label,path_list,origin_length) in tqdm(enumerate(train_loader)):
            image = image.to(DEVICE)
            handcrafted = handcrafted.to(DEVICE)
            label = label.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image,handcrafted) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    elif model_name == 'wav_mlp_smile':
        for batch_idx,(image,handcrafted,label,path_list,origin_length) in tqdm(enumerate(train_loader)):
            image = image.to(DEVICE)
            handcrafted = handcrafted.to(DEVICE)
            label = label.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image,handcrafted) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    elif model_name == 'wav_res_phrase_eggfusion_lstm':
        for batch_idx,(image,label,path_list,origin_length) in enumerate(train_loader):
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
    elif model_name == 'wav_bixception_phrase_eggfusion_lstm':
        for batch_idx,(image,label,path_list,origin_length) in enumerate(train_loader):
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
    elif model_name == 'wav_res_phrase_eggfusion_mmtm':
        for batch_idx,(image,label,path_list,origin_length) in enumerate(train_loader):
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
    elif model_name == 'wav_res_phrase_eggfusion_mmtm_bam':#wav_res_phrase_eggfusion_mmtm_nonlocal
        for batch_idx,(image,label,path_list,origin_length) in enumerate(train_loader):
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
    elif model_name == 'wav_res_phrase_eggfusion_mmtm_nonlocal':#
        for batch_idx,(image,label,path_list,origin_length) in enumerate(train_loader):
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
    elif model_name =='wav_res_time_attention':
    #8. 학습
        for batch_idx,(image,label,path_list,origin_length) in enumerate(train_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image,augment=False) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    elif model_name == 'wav_res_splicing':
        for batch_idx,(image,label_a,label_b,lam,path_list) in enumerate(train_loader):
            #splicing data가 넘어온다.
            image = image.to(DEVICE)
            label_a = label_a.to(DEVICE)
            label_b = label_b.to(DEVICE)
            lam = lam.to(DEVICE)
            # padding
            #image=pad_samples(image)

            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image) # model로 output을 계산
            
            loss = splicing_criterion(criterion,output, label_a,label_b,lam)
            #loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
           
            prediction = output.max(1,keepdim=True)[1].squeeze() # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만. 인덱스만 가져오면 당연히 안되고 각 확률을 합해줘야함.
            
             #여기 고치기. predictipn 계산이 잘못된것으로 보인다.
            correct += (lam * prediction.eq(label_a.data)).sum().float() + ((1 - lam) * prediction.eq(label_b.data)).sum().float()

            #correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    elif model_name == 'wav_res_mixup':
        for batch_idx,(image,label_a,label_b,lam,path_list) in enumerate(train_loader):
            #splicing data가 넘어온다.
            image = image.to(DEVICE)
            label_a = label_a.to(DEVICE)
            label_b = label_b.to(DEVICE)
            lam = lam
            # padding
            #image=pad_samples(image)

            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image) # model로 output을 계산
            loss = mixup_criterion(criterion, output, label_a, label_b, lam)
            #loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            same_label=label_a.eq(label_b).sum().float()
            total += (same_label + max([lam,1-lam])*(image.size(0)-same_label ) )
            #print("total: ", max([lam,1-lam])*image.size(0) )


            prediction = output.max(1,keepdim=True)[1].squeeze() # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            #print(prediction)
            probs = torch.softmax(output,dim=1)
            
            correct += (lam * prediction.eq(label_a.data)).sum().float() + ((1 - lam) * prediction.eq(label_b.data)).sum().float()
            #print("a :",(lam * prediction.eq(label_a.data)).sum().float());print("b : ",((1 - lam) * prediction.eq(label_b.data)).sum().float())
            #correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
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
    elif model_name == 'wav_res_concat_phrase_vowel':
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
    elif model_name == 'wav_res_concat_allfusion_attention':
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
    elif model_name == 'wav_res_latefusion_phrase_vowel':
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
        for batch_idx,(image,label,path_list,origin_length) in enumerate(train_loader):
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
    

    #믹스업 일때만 acc를 달리 계산
    if model_name=='wav_res_mixup':
        train_accuracy = 100. * correct / total
    else:
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
            for image,label,path_list,origin_length in valid_loader:
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
    elif model_name == 'wav_res_smile':
        with torch.no_grad():
            for image,handcrafted,label,path_list,origin_length in valid_loader:
                image = image.to(DEVICE)
                handcrafted = handcrafted.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image,handcrafted)
                #print(F.softmax(output))
                #print(output.size())
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #print(prediction.eq(label.view_as(prediction)))
                #print(path_list)
                #true.false값을 sum해줌. item#wav_res_smile
    elif model_name == 'wav_vgg16_smile':
        with torch.no_grad():
            for image,handcrafted,label,path_list,origin_length in valid_loader:
                image = image.to(DEVICE)
                handcrafted = handcrafted.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image,handcrafted)
                #print(F.softmax(output))
                #print(output.size())
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #print(prediction.eq(label.view_as(prediction)))
                #print(path_list)
                #true.false값을 sum해줌. item#wav_res_smile
    elif model_name == 'wav_mlp_smile':
        with torch.no_grad():
            for image,handcrafted,label,path_list,origin_length in valid_loader:
                image = image.to(DEVICE)
                handcrafted = handcrafted.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image,handcrafted)
                #print(F.softmax(output))
                #print(output.size())
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #print(prediction.eq(label.view_as(prediction)))
                #print(path_list)
                #true.false값을 sum해줌. item#wav_res_smile
    elif model_name == 'wav_res_phrase_eggfusion_lstm':
        with torch.no_grad():
            for image,label,path_list,origin_length in valid_loader:
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
    elif model_name == 'wav_bixception_phrase_eggfusion_lstm':
        with torch.no_grad():
            for image,label,path_list,origin_length in valid_loader:
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
    elif model_name == 'wav_res_phrase_eggfusion_mmtm':
        with torch.no_grad():
            for image,label,path_list,origin_length in valid_loader:
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
    elif model_name == 'wav_res_phrase_eggfusion_mmtm_nonlocal':
        with torch.no_grad():
            for image,label,path_list,origin_length in valid_loader:
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
    elif model_name == 'wav_res_time_attention':
        with torch.no_grad():
            for image,label,path_list,origin_length in valid_loader:
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
    elif model_name == 'wav_res_splicing':
        with torch.no_grad():
            for image,label,path_list,origin_length in valid_loader:
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
    elif model_name == 'wav_res_mixup':
        with torch.no_grad():
            for image,label,path_list,origin_length in valid_loader:
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
    elif model_name == 'wav_res_concat_phrase_vowel':
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
    elif model_name == 'wav_res_concat_allfusion_attention':
        with torch.no_grad():
            for image,label,path_list in valid_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #true.false값을 sum해줌. item
    elif model_name == 'wav_res_latefusion_phrase_vowel':
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
            for image,label,path_list,origin_length in valid_loader:
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
    valid_loss /= len(valid_loader.dataset)
    valid_accuracy = 100. * correct / len(valid_loader.dataset)
    return valid_loss,valid_accuracy

