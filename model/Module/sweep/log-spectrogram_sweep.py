
 
import wandb
import numpy as np

 
import wandb
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn # 인공 신경망 모델들 모아놓은 모듈
import torch.nn.functional as F #그중 자주 쓰이는것들을 F로
from torchvision import transforms, datasets
import cv2
from torchvision import transforms, datasets
import pandas as pd
import os
from glob import glob
import torchvision.models as models
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from Utils.pytorchtools import EarlyStopping # 상위 폴더에 추가된 모듈.
import torchaudio
#import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import librosa, librosa.display 
import matplotlib.pyplot as plt
import os
from glob import glob
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split # train , test 분리에 사용.
import os
import random #데이터 shuffle 사용
from glob import glob
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pickle
from torch.utils.data import Dataset, DataLoader



#args = None #글로벌 변수로 접근 위한 것



def main(args):
    #default param
    run_config = dict(
        # spectrum
        sr=16000,
        n_fft=2048,
        hop_length=50,
        win_length=300,
    )

    
    # n_mels
    name ="logspectrogram-seed"+str(args.seed)
    sweep_config = {
        "project" : "SVD-voice-disorder",
        "name": name,
    "method": "random",
    "parameters": {
            "n_fft": {
                    "values": np.arange(50,1000,10).tolist() # 제일 중요한 파라미터. 윈도우에서 몇만큼의 데이터를 사용하고 나머지 패딩할지. window + n_fft로 만들것
            },
            "win_length": {
                    "values": np.arange(50,4096,10).tolist()
                },
            "hop_length": {
                "values": np.arange(50,1000,50).tolist()
            },          
        }
    }

    sweep_id = wandb.sweep(sweep_config,project="SVD-hyp-sweep2",entity="bub3690")


    





    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    #DEVICE = torch.device('cpu')
    print('Using Pytorch version : ',torch.__version__,' Device : ',DEVICE)

    

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1234)


    # # SVD 문장 데이터에서 Feature 추출
    # - mfcc

    



    # # 데이터 나누기 - Stratified KFold
    # 
    # - pathology : 1194 / healthy : 634 / 총 1828
    # - k = 5


    # ## 1. test/ train 나누기
    # 

    
    speaker_data=pd.read_excel("../../../voice_data/only_organics_healthy_available.xlsx")

    
    pathology = speaker_data[speaker_data['PATHOLOGY']=='p']['SPEAKER'].unique().tolist()
    healthy = speaker_data[speaker_data['PATHOLOGY']=='n']['SPEAKER'].unique().tolist()
    print(len(pathology))
    print(len(healthy))

    
    #겹치는 speaker는 곱하기 100을 해준다.
    #3명이 겹친다.
    changed_patients = list(set(healthy) & set(pathology))

    for patient in changed_patients:
        temp=pathology[pathology.index(patient)]*100
        pathology[pathology.index(patient)] = temp
        

    
    #train test 나누기




    random_state = args.seed # 1004,1005,1006,1007,1008

    X = pathology+healthy # path 데이터 합
    print("총 데이터수 : ",len(X))
    Y = [] # 라벨
    for idx,x in enumerate(X):
        if idx<427:
            Y.append("pathology")
        else:
            Y.append("healthy")

    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y, random_state=random_state) #456
    #stratify를 넣어서, test에도 라벨별 잘 분류되게 한다.

    print("---")
    print("훈련 셋 : ",len(Y),Counter(Y))
    print("테스트 셋 : ",len(Y_test),Counter(Y_test))
    print("---")




    # ## 2. stratified k-fold

    
    #1. train, test 나누기
    #stratified kfold



    skf_iris = StratifiedKFold(n_splits=5,shuffle=True,random_state=456)
    cnt_iter = 0

    X_train_list = [] #데이터 셋 보관
    Y_train_list = []

    X_valid_list = []
    Y_valid_list = []

    for train_idx, test_idx in skf_iris.split(X,Y):
        
        #split으로 반환된 인덱스를 이용하여, 학습 검증용 테스트 데이터 추출
        cnt_iter += 1
        X_train, X_valid = [X[idx] for idx in train_idx.tolist() ], [X[idx] for idx in test_idx.tolist() ]
        Y_train, Y_valid = [Y[idx] for idx in train_idx.tolist() ], [Y[idx] for idx in test_idx.tolist() ]
        
        X_train_list.append(X_train)
        X_valid_list.append(X_valid)
        
        Y_train_list.append(Y_train)
        Y_valid_list.append(Y_valid)
        
        
        #학습 및 예측
        
        label_train = Y_train
        label_test = Y_valid
        unique_train, train_counts = np.unique(label_train, return_counts = True)
        unique_test, test_counts = np.unique(label_test, return_counts = True)
        
        uniq_cnt_train = dict(zip(unique_train, train_counts))
        uniq_cnt_test = dict(zip(unique_test, test_counts))
        
        
        
        print('교차 검증 : {}'.format(cnt_iter))
        print('학습 레이블 데이터 분포 : \n', uniq_cnt_train)
        print('검증 레이블 데이터 분포 : \n', uniq_cnt_test,'\n')




    # # speaker to voice

    
    # speaker to voice

    all_train_record_list = []
    all_valid_record_list = []
    all_test_record_list = []

    all_train_label_list = []
    all_valid_label_list = []
    all_test_label_list = []

    #train
    for fold_idx,fold in enumerate(X_train_list):
        fold_record=[]
        fold_y_record=[]
        for idx,speaker in enumerate(fold):
            record_list = speaker_data[speaker_data['SPEAKER']==speaker]['RECORDING'].tolist()
            label_list = [ Y_train_list[fold_idx][idx] ] * len(record_list)
            
            fold_record += record_list
            fold_y_record += label_list
        all_train_record_list.append(fold_record)
        all_train_label_list.append(fold_y_record)

        
    #valid
    for fold_idx,fold in enumerate(X_valid_list):
        fold_record=[]
        fold_y_record=[]
        for idx,speaker in enumerate(fold):
            record_list = speaker_data[speaker_data['SPEAKER']==speaker]['RECORDING'].tolist()
            label_list = [ Y_valid_list[fold_idx][idx] ] * len(record_list)
            
            fold_record += record_list
            fold_y_record += label_list
        all_valid_record_list.append(fold_record)
        all_valid_label_list.append(fold_y_record)
        
    #test
    fold_record=[]
    fold_y_record=[]
    for idx,speaker in enumerate(X_test):
        record_list = speaker_data[speaker_data['SPEAKER']==speaker]['RECORDING'].tolist()
        label_list = [ Y_test[idx] ] * len(record_list)
        fold_record += record_list
        fold_y_record += label_list
    all_test_record_list = fold_record
    all_test_label_list = fold_y_record


    X_train_list = all_train_record_list
    X_valid_list = all_valid_record_list
    X_test = all_test_record_list

    Y_train_list = all_train_label_list
    Y_valid_list = all_valid_label_list
    Y_test = all_test_label_list




    # ## 3. random over sampling

    
    #2. random over sampling
    for i in range(5):
        X_temp = np.array(X_train_list[i]).reshape(-1,1)#각 데이터를 다 행으로 넣음. (1194,1)
        #Y = np.array(Y)
        ros = RandomOverSampler(random_state = 123)
        X_res,Y_res = ros.fit_resample(X_temp,Y_train_list[i])
        
        print("\n fold{} ".format(i))
        print('before dataset shape {}'.format(Counter(Y_train_list[i])) )
        print('Resampled dataset shape {}'.format(Counter(Y_res)) )   
        
        #원래대로 돌리기
        X_res=X_res.reshape(1, -1)
        X_train_list[i]=X_res[0].tolist()
        Y_train_list[i]=Y_res

    



        
    #load
    with open("../../../voice_data/organics/phrase_sig_dict.pickle","rb") as fr:
        phrase_dict = pickle.load(fr)

    #with open("../../voice_data/organics/phrase_minmax_scaler_hyper.pickle","rb") as fr:
    #    phrase_scaler = pickle.load(fr)
        


    # # 데이터 정의
    # - 추가적으로 데이터의 크기를 맞춰주기 위해 3초로 padding 및 truncate 실시 https://sequencedata.tistory.com/25 FixAudioLength
    # - 논문에서는 400frame으로 설정.(여기서는 500frame)
    # - 전처리 방법 결정.
    # 

    
    # 데이터 로더

    

    classes = ["healthy","pathology"]


    class svd_dataset(Dataset):
        def __init__(self,data_path_list,classes,data_num,training,spectro_params,transform=None,normalize=None):
            #클래스에서 사용할 인자를 받아 인스턴스 변수로 저장하는 일을 한다.
            #예를들면, 이미지의 경로 리스트를 저장하는 일을 하게 된다.
            
            #data_num : k 개 데이터 셋 중 어떤것을 쓸지
            #test인지 아닌지.
            
            self.path_list = data_path_list[data_num]
            self.data_num = data_num
            self.training = training
            self.label = svd_dataset.get_label(self.path_list,training,data_num)
            self.classes=classes
            self.transform=transform
            self.normalize=normalize
            
            # sweep params
            self.spectro_params = spectro_params
            # sr,win_length,n_fft,hop_length
            
        
        @classmethod
        def get_label(cls,data_path_list,training,data_num):
            label_list=[]
            
            if training:
                for idx,x in enumerate(data_path_list):
                    label_list.append(Y_train_list[data_num][idx])
            else:
                for idx,x in enumerate(data_path_list):
                    label_list.append(Y_valid_list[data_num][idx])
            #print(label_list)
            return label_list
        
        
        def __len__(self):
            return len(self.path_list)
            #데이터 셋의 길이를 정수로 반환한다.     
        
        
        def __getitem__(self, idx):
            """
            """
            sig = phrase_dict[ str(self.path_list[idx])+'-phrase.wav'] 
            #sig = preemphasis(sig)
            origin_length = sig.shape[0]
            
            #if sig.shape[0] > self.mel_params["sr"]*2:
            #    origin_length = self.mel_params["sr"]*2
            
            #origin_frame_size = 1 + int(np.floor(origin_length//self.mel_params["hop_length"]))
            
            length = self.spectro_params["sr"]*2 #sample rate *2 padding을 위한 파라미터 (하이퍼 파라미터로인해 사이즈는 계속 바뀐다.)
            pad1d = lambda a, i: a[0:i] if a.shape[0] > i else np.hstack((a, np.zeros((i-a.shape[0]))))        
            sig = pad1d(sig,length)        
            
            ###signal norm
            sig = (sig-sig.mean())/sig.std()
            ###

            stft = librosa.stft(y=sig,
                            win_length=self.spectro_params["win_length"],
                            n_fft=self.spectro_params["win_length"]+self.spectro_params["n_fft"],
                            hop_length=self.spectro_params["hop_length"]
                            )            
            magnitude = np.abs(stft)
            log_spectrogram = librosa.amplitude_to_db(magnitude)
            
            if self.transform:
                log_spectrogram = self.transform(log_spectrogram).type(torch.float32)# 데이터 0~1 정규화
                MSF = torch.stack([log_spectrogram, log_spectrogram, log_spectrogram])# 3채널로 복사.
                MSF = MSF.squeeze(dim=1)    
                
                # global normalize
                if self.normalize:
                    #MFCCs=self.normalize(MFCCs)
                    MSF = self.normalize(MSF)
            else:
                pass
                #print("else")
                mel_feature = torch.from_numpy(mel_feature).type(torch.float32)
                mel_feature=mel_feature.unsqueeze(0)#cnn 사용위해서 추가
                #MFCCs = MFCCs.permute(2, 0, 1)
            return MSF, self.classes.index(self.label[idx])
        


    # # 데이터로더

    
    #3. 하이퍼 파라미터
    BATCH_SIZE =  16 #한 배치당 32개 음성데이터
    EPOCHS = 50 # 전체 데이터 셋을 50번 반복
    lr=1e-4
    augment_kind="no"
    weight_decay = 0


    # # RESNET

    
    # 모델
    # pretrained

    class ResLayer(nn.Module):
        def __init__(self):
            super(ResLayer, self).__init__()
            self.model = models.resnet18(pretrained=True).cuda() 
            self.num_ftrs = self.model.fc.out_features
            
            
            
            self.fc = nn.Sequential(       
                nn.Linear(self.num_ftrs, 64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(64,50),
                                nn.BatchNorm1d(50),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(50,2)
                                )
            

        def forward(self, x):
            x = self.model(x)
            x  = self.fc(x)
            return x





    def model_initialize():
        model = ResLayer().cuda()
        return model

    model=model_initialize()

    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    print(model)

    
    # get the model summary
    #summary(model, input_size=(3, 128, 300), device=DEVICE.type)

    
    #8. 학습
    def train(model,train_loader,optimizer, log_interval):
        model.train()
        correct = 0
        train_loss = 0
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
    def evaluate(model,valid_loader):
        model.eval()
        valid_loss = 0
        correct = 0
        #no_grad : 그래디언트 값 계산 막기.
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


    
    #데이터 로더 제작 함수

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def load_data(data_ind,num_workers):

        train_loader = torch.utils.data.DataLoader(dataset = 
                                                svd_dataset(
                                                    X_train_list,
                                                    classes,
                                                    transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                    data_num=data_ind,
                                                    #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                                    #mfcc_normalize=(53.5582, 217.43),
                                                    spectro_params = dict(
                                                            sr = wandb.config.sr,
                                                            n_fft = wandb.config.n_fft,
                                                            win_length = wandb.config.win_length,
                                                            hop_length = wandb.config.hop_length,
                                                    ),
                                                    training=True
                                                ),
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                worker_init_fn=seed_worker,
                                                num_workers=num_workers
                                                ) # 순서가 암기되는것을 막기위해.


        validation_loader = torch.utils.data.DataLoader(dataset = 
                                                svd_dataset(
                                                    X_valid_list,
                                                    classes,
                                                    transform = transforms.ToTensor(),
                                                    data_num=data_ind,
                                                    #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                                    #mfcc_normalize=(53.5582, 217.43),
                                                    spectro_params = dict(
                                                            sr = wandb.config.sr,
                                                            n_fft = wandb.config.n_fft,
                                                            win_length = wandb.config.win_length,
                                                            hop_length = wandb.config.hop_length,
                                                    ),
                                                    training=False
                                                ),
                                                        batch_size = BATCH_SIZE,
                                                        shuffle = True,
                                                        worker_init_fn=seed_worker,
                                                        num_workers=num_workers) 
        return train_loader,validation_loader



    # # 학습

    
    #10. 학습 및 평가.
    # resnet34 pretrained true
    # kfold 적용

    train_accs = []
    valid_accs = []



    def all_train(args):
        wandb.init(project="SVD-hyp-sweep2", entity="bub3690",config=run_config)
        data_ind = 1
        check_path ='../checkpoint/log-spectrogram_sweep_'+str(args.seed)+'_organics_speaker.pt'
        print(check_path)
        #wandb.run.name = 'n'### 여기 수정 ###
        print("config:", dict(wandb.config))    

        early_stopping = EarlyStopping(patience = 5, verbose = True, path=check_path)
        train_loader,validation_loader = load_data(data_ind-1,args.num_workers)

        best_train_acc = 0 # accuracy 기록용
        best_valid_acc = 0
        
        best_train_loss = 0
        best_valid_loss = 0

        model=model_initialize()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)


        print("[{} 교차검증] 학습 시작\n ----- ".format(data_ind))
        for Epoch in range(1,EPOCHS+1):
            train_loss,train_accuracy=train(model,train_loader,optimizer,log_interval=31)
            valid_loss,valid_accuracy = evaluate(model, validation_loader)

            print("\n[EPOCH:{}]\t Train Loss:{:.4f}\t Train Acc:{:.2f} %  | \tValid Loss:{:.4f} \tValid Acc: {:.2f} %\n".
                format(Epoch,train_loss,train_accuracy,valid_loss,valid_accuracy))
            #wandb.log({"metric": run.config.param1, "epoch": epoch})
                
            early_stopping(valid_loss, model)
            if -early_stopping.best_score == valid_loss:
                best_train_acc, best_valid_acc = train_accuracy,valid_accuracy
                best_train_loss, best_valid_loss = train_loss,valid_loss
                
                wandb.log({"Valid/Loss": best_valid_loss, 
                        "Valid/Accuracy": best_valid_acc,
                        }, step=Epoch)
                #wandb.run.summary.update({"best_valid_{}fold_acc".format(data_ind) : best_valid_acc})
            else:
                # 이전 최고 기록을 log
                wandb.log({"Valid/Loss": best_valid_loss, 
                        "Valid/Accuracy": best_valid_acc,
                        }, step=Epoch)

            if early_stopping.early_stop:
                    train_accs.append(best_train_acc)
                    valid_accs.append(best_valid_acc)
                    #여기 최고기록만 갱신하면 5fold 가능.
                    
                    print("[{} 교차검증] Early stopping".format(data_ind))
                    break

            if Epoch==EPOCHS:
                #만약 early stop 없이 40 epoch라서 중지 된 경우.
                train_accs.append(best_train_acc)
                valid_accs.append(best_valid_acc)

    
    wandb.agent(sweep_id, function=lambda: all_train(args),count=100)



if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Voice Disorder Detection sweep')
    parser.add_argument('--seed',type=int,default=1004,help='set the test seed')
    parser.add_argument('--num-workers',type=int,default=0,help='set the workers')


    args = parser.parse_args()

    main(args)

