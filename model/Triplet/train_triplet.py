
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
from tqdm import tqdm
from Utils.pytorchtools import EarlyStopping # 상위 폴더에 추가된 모듈.

import torchaudio
import torchaudio.transforms as T

import numpy as np
import librosa, librosa.display 
import matplotlib.pyplot as plt

# classifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split # train , test 분리에 사용.
import os
import random #데이터 shuffle 사용
from glob import glob
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import pickle

wandb.init(project="SVD-voice-disorder", entity="bub3690",settings=wandb.Settings(_disable_stats=True))
wandb.run.name = 'triplet-organics-speaker-original-1004'
wandb.run.save()


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
#DEVICE = torch.device('cpu')
print('Using Pytorch version : ',torch.__version__,' Device : ',DEVICE)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1234)






#default param
mfcc_run_config = dict(
    sr=16000,
    n_mfcc=27,
    #dct_type=3, # type2 default
    lifter = 35,

    
    #mel spectro
    n_mels=170,
    hop_length=750,
    n_fft =14056,    
    win_length=1100,
    f_max=8000,
    
    # training
    #batch_size=32,
    mel_scale ='htk',
    
    # data
    fold=1,
)

mel_run_config = dict(
    sr=16000,
    n_mels=128,
    win_length =  300,
    n_fft= 2048,
    hop_length= 50,
    f_max = 8000    
)


spectro_run_config =dict(
    sr=16000,
    n_fft=350,
    hop_length=50,
    win_length=350,
    # training
    batch_size=16,
)




#또는 10ms만큼으로 한다고 한다.
#hop_length가 mfcc의 frame수를 결정한다.


# # 데이터 나누기 - Stratified KFold
# 
# - pathology : 1194 / healthy : 634 / 총 1828
# - k = 5


# ## 1. test/ train 나누기
# 


speaker_data=pd.read_excel("../../voice_data/only_organics_healthy_available_ver2.xlsx")


pathology = speaker_data[speaker_data['PATHOLOGY']=='p']['SPEAKER'].unique().tolist()
healthy = speaker_data[speaker_data['PATHOLOGY']=='n']['SPEAKER'].unique().tolist()
print(len(pathology))
print(len(healthy))


list(set(healthy) & set(pathology))


#겹치는 speaker는 곱하기 100을 해준다.
#겹치는 speaker는 그대로 둔다.

changed_patients = list(set(healthy) & set(pathology))

for patient in changed_patients:
    temp=pathology[pathology.index(patient)]*100
    pathology[pathology.index(patient)] = temp
    


pathology[pathology.index(152400)]


#train test 나누기







random_state = 1004 # 1004,1005,1006,1007,1008

X = pathology+healthy # path 데이터 합
print("총 데이터수 : ",len(X))
Y = [] # 라벨
for idx,x in enumerate(X):
    if idx<426:
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

label_changer = dict({"healthy":"n","pathology":"p"})


all_train_record_list = []
all_valid_record_list = []
all_test_record_list = []

all_train_label_list = []
all_valid_label_list = []
all_test_label_list = []

print("train. speaker to voice")
#train
for fold_idx,fold in enumerate(X_train_list):
    fold_record=[]
    fold_y_record=[]
    for idx,speaker in enumerate(fold):
        record_list = speaker_data[ (speaker_data['SPEAKER']==speaker) & (speaker_data['PATHOLOGY']==label_changer[Y_train_list[fold_idx][idx]])]['RECORDING'].tolist()
        if record_list == []:
            # speaker가 healthy, pathology 모두 있는 경우
            #print(speaker)
            speaker = speaker // 100
            record_list = speaker_data[(speaker_data['SPEAKER']==speaker) & (speaker_data['PATHOLOGY']==label_changer[Y_train_list[fold_idx][idx]] ) ]['RECORDING'].tolist()
            print(record_list)

        label_list = [ Y_train_list[fold_idx][idx] ] * len(record_list)       
        fold_record += record_list
        fold_y_record += label_list
    all_train_record_list.append(fold_record)
    all_train_label_list.append(fold_y_record)

print("valid. speaker to voice")
#valid
for fold_idx,fold in enumerate(X_valid_list):
    fold_record=[]
    fold_y_record=[]
    for idx,speaker in enumerate(fold):
        record_list = speaker_data[ (speaker_data['SPEAKER']==speaker) & (speaker_data['PATHOLOGY']==label_changer[Y_valid_list[fold_idx][idx]]) ]['RECORDING'].tolist()
        if record_list == []:
            # speaker가 healthy, pathology 모두 있는 경우
            #print(speaker)
            speaker = speaker // 100
            record_list = speaker_data[(speaker_data['SPEAKER']==speaker) & (speaker_data['PATHOLOGY']==label_changer[Y_valid_list[fold_idx][idx]] ) ]['RECORDING'].tolist()
            print(record_list)
        label_list = [ Y_valid_list[fold_idx][idx] ] * len(record_list)
        
        fold_record += record_list
        fold_y_record += label_list
    all_valid_record_list.append(fold_record)
    all_valid_label_list.append(fold_y_record)

print("test. speaker to voice")
#test
fold_record=[]
fold_y_record=[]
for idx,speaker in enumerate(X_test):
    record_list = speaker_data[(speaker_data['SPEAKER']==speaker) & (speaker_data['PATHOLOGY']==label_changer[Y_test[idx]] )]['RECORDING'].tolist()
    if record_list == []:
        # speaker가 healthy, pathology 모두 있는 경우
        #print(speaker)
        speaker = speaker // 100
        record_list = speaker_data[(speaker_data['SPEAKER']==speaker) & (speaker_data['PATHOLOGY']==label_changer[Y_test[idx]] ) ]['RECORDING'].tolist()
        print(record_list)
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
with open("../../voice_data/organics_ver2/phrase_dict_ver2.pickle","rb") as fr:
    phrase_dict = pickle.load(fr)

#with open("../../voice_data/organics/phrase_minmax_scaler_hyper.pickle","rb") as fr:
#    phrase_scaler = pickle.load(fr)
    


# # 데이터 정의
# - 추가적으로 데이터의 크기를 맞춰주기 위해 3초로 padding 및 truncate 실시 https://sequencedata.tistory.com/25 FixAudioLength
# - 논문에서는 400frame으로 설정.(여기서는 500frame)
# - 전처리 방법 결정.
# 


# 데이터 로더


#default param
mfcc_run_config = dict(
    sr=16000,
    n_mfcc=27,
    #dct_type=3, # type2 default
    lifter = 35,

    
    #mel spectro
    n_mels=170,
    hop_length=750,
    n_fft =14056,    
    win_length=1100,
    f_max=8000,
    
    # training
    #batch_size=32,
    mel_scale ='htk',
    
    # data
    fold=1,
)

mel_run_config = dict(
    sr=16000,
    n_mels=128,
    win_length =  300,
    n_fft= 2048,
    hop_length= 50,
    f_max = 8000    
)


spectro_run_config =dict(
    sr=16000,
    n_fft=350,
    hop_length=50,
    win_length=350,
    # training
    batch_size=16,
)


# score based online sampling


classes = ["healthy","pathology"]

probs=0

class svd_dataset_wav_hard(Dataset):
    def __init__(self,
                data_path_list,
                y_label_list,
                classes,
                mel_params,
                dataset='phrase',
                transform=None,
                is_train=False,):

        #클래스에서 사용할 인자를 받아 인스턴스 변수로 저장하는 일을 한다.
        #예를들면, 이미지의 경로 리스트를 저장하는 일을 하게 된다.
        
        #data_num : k 개 데이터 셋 중 어떤것을 쓸지
        #test인지 아닌지.
        
        self.path_list = data_path_list
        self.label = y_label_list # label data
        self.classes=classes
        self.transform=transform

        self.is_train = is_train

        # sweep params
        self.mel_params = mel_params


        self.dataset=dataset

            
    def __len__(self):
        return len(self.path_list)
        #데이터 셋의 길이를 정수로 반환한다.     
    
    def get_score(anchor,candidate,alpha,beta):
        age_diff = np.abs(anchor['AGE'].item() - candidate['AGE'].item()).item()
        gender_diff = int(anchor['GENDER'].item()==candidate['GENDER'].item())
        normal_criterion = int((anchor['RECORDING'].item() == candidate['RECORDING'].item()) | (anchor['SPEAKER'].item() == candidate['SPEAKER'].item()))
        score = alpha*age_diff + beta*gender_diff - 1000*normal_criterion
        return score

    def __getitem__(self, idx):
        """
        WAV 파일을 읽어서, MODEL에 전달.
        """
        # print(str(self.path_list[idx])+'-'+self.dataset+'.wav' )
        anchor_sig = phrase_dict[ str(self.path_list[idx])+'-'+self.dataset+'.wav' ] 
        anchor_label = self.label[idx]

        if self.is_train:
            
            hard_mining = random.choices([True,False],weights=[probs,1-probs],k=1)[0]
            
            if hard_mining:
                
                anchor_speaker = speaker_data.where(speaker_data['RECORDING']==self.path_list[idx]).dropna()['SPEAKER'].item()
                anchor_df = speaker_data[speaker_data['RECORDING']==self.path_list[idx]]
                check_positive = True
                check_negative = True

                positive_index_list = [ind for ind,a in enumerate(self.label) if a == anchor_label]
                negative_index_list = [ind for ind,a in enumerate(self.label) if a != anchor_label]
                
                positive_candidate = np.random.choice(positive_index_list, 50, replace=False).tolist()
                negative_candidate = np.random.choice(negative_index_list, 50, replace=False).tolist()
                
                positive_candidate_score = [ svd_dataset_wav_hard.get_score(anchor_df, speaker_data[ speaker_data['RECORDING'] == self.path_list[pos]],0.5,-10 ) for pos in positive_candidate]
                negative_candidate_score = [ svd_dataset_wav_hard.get_score(anchor_df, speaker_data[ speaker_data['RECORDING'] == self.path_list[neg]],-0.5,+10 ) for neg in negative_candidate]


                positive_index = np.argsort(positive_candidate_score)[::-1][0]
                negative_index = np.argsort(negative_candidate_score)[::-1][0]
                
                positive_sig = phrase_dict[ str(self.path_list[positive_candidate[positive_index]])+'-'+self.dataset+'.wav' ]
                #print("pos : ",self.path_list[idx], str(self.path_list[positive_candidate[positive_index]]) )
                negative_sig = phrase_dict[ str(self.path_list[negative_candidate[negative_index]])+'-'+self.dataset+'.wav' ]
                #print("neg : ",self.path_list[idx],str(self.path_list[negative_candidate[negative_index]]))
                
            else:
                
                anchor_speaker = speaker_data.where(speaker_data['RECORDING']==self.path_list[idx]).dropna()['SPEAKER'].item()
                check_positive = True
                check_negative = True

                positive_index_list = [ind for ind,a in enumerate(self.label) if a == anchor_label]
                negative_index_list = [ind for ind,a in enumerate(self.label) if a != anchor_label]
                

                while check_positive:
                    positive_index = random.choice(positive_index_list)
                    positive_sample = speaker_data[ speaker_data['RECORDING']==self.path_list[positive_index]]

                    # 확률을 줘서, 나이,성별이 차이가 나는 것을 뽑기?

                    if ((positive_sample['RECORDING'] != self.path_list[idx]) & (positive_sample['SPEAKER'] != anchor_speaker)).item():
                        check_positive=False
                positive_sig = phrase_dict[ str(self.path_list[positive_index])+'-'+self.dataset+'.wav' ]


                #시나리오를 내거티브에서 남자. 남자. 노인 노인 매칭되게 하는건 어떨까?
                while check_negative:
                    negative_index = random.choice(negative_index_list)
                    negative_sample = speaker_data[speaker_data['RECORDING']==self.path_list[negative_index]]

                    # 확률을 줘서, 나이,성별이 비슷한 것 뽑기. 또는 미니 배치내에서 가장 점수 높은 것을 선정.

                    if ((negative_sample['RECORDING'] != self.path_list[idx]) & (negative_sample['SPEAKER'] != anchor_speaker)).item():
                        check_negative=False
                negative_sig = phrase_dict[ str(self.path_list[negative_index])+'-'+self.dataset+'.wav' ]         

            origin_length = anchor_sig.shape[0]
            if anchor_sig.shape[0] > self.mel_params["sr"]*3:
                origin_length = self.mel_params["sr"]*3
            
            origin_frame_size = 1 + int(np.floor(origin_length//self.mel_params["hop_length"]))
            
            length = self.mel_params["sr"]*3 #sample rate *2 padding을 위한 파라미터 (하이퍼 파라미터로인해 사이즈는 계속 바뀐다.)
            pad1d = lambda a, i: a[0:i] if a.shape[0] > i else np.hstack((a, np.zeros((i-a.shape[0]))))

            anchor_sig = pad1d(anchor_sig,length)
            positive_sig = pad1d(positive_sig,length)
            negative_sig = pad1d(negative_sig,length)
            
            ###signal norm
            # anchor_sig = (anchor_sig-anchor_sig.mean())/anchor_sig.std()
            # positive_sig = (positive_sig-positive_sig.mean())/positive_sig.std()
            # negative_sig = (negative_sig-negative_sig.mean())/negative_sig.std()
            ###

            anchor_sig=torch.from_numpy(anchor_sig).type(torch.float32)# 타입 변화
            positive_sig=torch.from_numpy(positive_sig).type(torch.float32)# 타입 변화
            negative_sig=torch.from_numpy(negative_sig).type(torch.float32)# 타입 변화

            anchor_sig=anchor_sig.unsqueeze(0)
            positive_sig=positive_sig.unsqueeze(0)
            negative_sig=negative_sig.unsqueeze(0)
            
            return anchor_sig, positive_sig, negative_sig, self.classes.index(self.label[idx]), str(self.path_list[idx])
        else:
            length = self.mel_params["sr"]*3 #sample rate *2 padding을 위한 파라미터 (하이퍼 파라미터로인해 사이즈는 계속 바뀐다.)
            pad1d = lambda a, i: a[0:i] if a.shape[0] > i else np.hstack((a, np.zeros((i-a.shape[0]))))

            anchor_sig = pad1d(anchor_sig,length)
            anchor_sig=torch.from_numpy(anchor_sig).type(torch.float32)# 타입 변화
            anchor_sig=anchor_sig.unsqueeze(0)
            return anchor_sig, self.classes.index(self.label[idx]), str(self.path_list[idx])



#3. 하이퍼 파라미터
BATCH_SIZE =  16 #한 배치당 32개 음성데이터
EPOCHS = 50 # 전체 데이터 셋을 50번 반복
lr=1e-4
augment_kind="no"
weight_decay = 0
probs=probs


wandb.config.update({
    "learning_rate": lr,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "augment":augment_kind,
    "weight_decay":weight_decay,
    "margin":1,
    "probs":probs,
    "특이사항":"triplet loss",
})


#DATA LOADER 함수가 BATCH_size 단위로 분리해 지정.

#확인을 위해 데이터셋 하나만 확인
dataset = 'phrase'

train_loader = DataLoader(dataset = svd_dataset_wav_hard(
                                            X_train_list[0],
                                            Y_train_list[0],
                                            classes,
                                            mel_params = mel_run_config,
                                            transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                            is_train = True,
                                            dataset= dataset
                                        ),
                                        batch_size = BATCH_SIZE,
                                        shuffle = True,
                                        #worker_init_fn=seed_worker
                                        ) # 순서가 암기되는것을 막기위해.

validation_loader = DataLoader(dataset = 
                                        svd_dataset_wav_hard(
                                            X_valid_list[0],
                                            Y_valid_list[0],
                                            classes,
                                            mel_params = mel_run_config,
                                            transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                            dataset= dataset
                                        ),
                                        batch_size = BATCH_SIZE,
                                        shuffle = True,
                                        #worker_init_fn=seed_worker
                                        )



X_train_list[0][8]


# 테스트 데이터 로더.
test_loader = DataLoader(dataset = 
                                        svd_dataset_wav_hard(
                                            X_test,
                                            Y_test,
                                            classes,
                                            mel_params = mel_run_config,
                                            transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                            dataset= dataset
                                        ),
                                        batch_size = BATCH_SIZE,
                                        shuffle = True,
                                        num_workers=0
                                        #worker_init_fn=seed_worker
                                        )



# # 데이터 확인



sr=16000
win_length =  mel_run_config["win_length"] # 400
n_fft= mel_run_config["n_fft"] # WINDOWS SIZE중 사용할 길이. WINDOW SIZE가 넘어가면 나머지 것들은 zero padding. 세로 길이
hop_length=mel_run_config["hop_length"] #  얼마만큼 시간 주기(sample)를 이동하면서 분석을 할 것인지. 일반적으로 window size의 1/4
#또는 10ms만큼으로 한다고 한다.
#hop_length가 mfcc의 frame수를 결정한다.

spectro_win_length =  spectro_run_config["win_length"] # 400
spectro_n_fft= spectro_run_config["n_fft"] # WINDOWS SIZE중 사용할 길이. WINDOW SIZE가 넘어가면 나머지 것들은 zero padding. 세로 길이
spectro_hop_length= spectro_run_config["hop_length"] #  얼마만큼 시간 주기(sample)를 이동하면서 분석을 할 것인지. 일반적으로 window size의 1/4




## 4. 데이터 확인하기
for (X_train,X_train_pos,X_train_neg,Y_train,_) in train_loader:
    print("X_train : ",X_train.size(),'type:',X_train.type())
    print("Y_train : ",Y_train.size(),'type:',Y_train.type())
    break

print(Y_train[0])
print(X_train[0])
#batch: 32 / 3채널 / frame수: 500  /  feature수: 13


#valiation set 확인
for (X_valid,Y_valid,_) in validation_loader:
    print("X_valid : ",X_valid.size(),'type:',X_valid.type())
    print("Y_valid : ",Y_valid.size(),'type:',Y_valid.type())
    break
print(X_valid[0])
print(Y_valid[0])

#batch: 32 / 3채널 / frame수: 500  /  feature수: 13


#test set 확인
for (test_data,test_label,_) in test_loader:
    print("X_test : ",test_data.size(),'type:',test_data.type())
    print("Y_test : ",test_label.size(),'type:',test_label.type())
    break

print(test_data[0])
print(test_label[0])

#batch: 32 / 3채널 / frame수: 500  /  feature수: 13


# # Resnet18 + triplet loss


class ResLayer(nn.Module):
    def __init__(self,emb_dim=128):
        super(ResLayer, self).__init__()
        self.model = models.resnet18(pretrained=True).cuda() 
        self.num_ftrs = self.model.fc.out_features
        self.emb_dim = emb_dim
        self.mel_scale = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=128,
            f_min=0,
            f_max=8000,
            center=True,
            pad_mode="constant",
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            window_fn=torch.hann_window
        )
        
        
        # self.fc = nn.Sequential(       
        #     nn.Linear(self.num_ftrs, 64),
        #                      nn.BatchNorm1d(64),
        #                      nn.ReLU(),
        #                      #nn.Dropout(p=0.5),
        #                      nn.Linear(64,50),
        #                      nn.BatchNorm1d(50),
        #                      nn.ReLU(),
        #                      #nn.Dropout(p=0.5),
        #                      nn.Linear(50,self.emb_dim)
        #                     )
        self.fc = nn.Sequential(       
            nn.Linear(self.num_ftrs, 64),
                             nn.BatchNorm1d(64),
                             nn.ReLU(),
                             #nn.Dropout(p=0.5),
                             nn.Linear(64,50),
                             nn.BatchNorm1d(50),
                             nn.ReLU(),
                             #nn.Dropout(p=0.5),
                             nn.Linear(50,self.emb_dim)
                            )
        

    def forward(self, x ,tsne=False):
        x = self.mel_scale(x)
        x = torchaudio.functional.amplitude_to_DB(x,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(x)) )
        x = x.squeeze()

        x = torch.stack([x,x,x],axis=1)
        
        x = self.model(x)

        if tsne:
            return x
        x  = self.fc(x)
        return x

def model_initialize(emb_dim=128):
    model = ResLayer(emb_dim).cuda()
    return model

emb_dim=128
model=model_initialize(emb_dim)


res=model(torch.randn(4,1,32000).to(DEVICE))
res.size()


criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
print(model)


#8. 학습
def train(model,train_loader,optimizer, log_interval):
    model.train()
    correct = 0
    train_loss = 0

    for batch_idx,(x_anchor,x_pos,x_neg,label,data_path) in enumerate(train_loader):
        x_anchor = x_anchor.to(DEVICE)
        x_pos = x_pos.to(DEVICE)
        x_neg = x_neg.to(DEVICE)
        label = label.to(DEVICE)
        #데이터들 장비에 할당
        optimizer.zero_grad() # device 에 저장된 gradient 제거
        anchor_out = model(x_anchor)
        positive_out = model(x_pos)
        negative_out = model(x_neg)

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss/=len(train_loader.dataset)
    return train_loss



#9. 학습 진행하며, validation 데이터로 모델 성능확인
def evaluate(model,valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    #no_grad : 그래디언트 값 계산 막기.
    with torch.no_grad():
        for image,label,_ in valid_loader:
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


def load_data(data_ind):

    #DATA LOADER 함수가 BATCH_size 단위로 분리해 지정.

    #확인을 위해 데이터셋 하나만 확인
    dataset = 'phrase'

    train_loader = DataLoader(dataset = svd_dataset_wav_hard(
                                                X_train_list[data_ind],
                                                Y_train_list[data_ind],
                                                classes,
                                                mel_params = mel_run_config,
                                                transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                is_train = True,
                                                dataset= dataset
                                            ),
                                            batch_size = BATCH_SIZE,
                                            shuffle = True,
                                            #worker_init_fn=seed_worker
                                            ) # 순서가 암기되는것을 막기위해.

    validation_loader = DataLoader(dataset = 
                                            svd_dataset_wav_hard(
                                                X_valid_list[data_ind],
                                                Y_valid_list[data_ind],
                                                classes,
                                                mel_params = mel_run_config,
                                                transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                dataset= dataset
                                            ),
                                            batch_size = BATCH_SIZE,
                                            shuffle = True,
                                            #worker_init_fn=seed_worker
                                            )
    return train_loader,validation_loader




##### 10. 학습 및 평가.
# resnet34 pretrained true
# kfold 적용

###
# train 과정
train_result_list = []
train_label_list = []
train_path_list = []
###


###
# valid 과정
valid_result_list = []
valid_label_list = []
valid_path_list = []
valid_probs_list = []
##






train_accs = []
valid_accs = []
EPOCHS = 300
for data_ind in range(1,2): 

    check_path = './checkpoint/checkpoint_triplet_ros_'+str(data_ind)+'_organics_speaker.pt'
    print(check_path)
    early_stopping = EarlyStopping(patience = 5, verbose = True, path=check_path)
    train_loader,validation_loader = load_data(data_ind-1)

    best_train_acc=0 # accuracy 기록용
    best_valid_acc=0
    best_test_acc=0
    
    model=model_initialize(emb_dim)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)    
    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
    #                                               max_lr=0.0001,
    #                                               steps_per_epoch=len(train_loader),
    #                                               epochs=20,
    #                                               anneal_strategy='linear')
    
    
    print("[{} 교차검증] 학습 시작\n ----- ".format(data_ind))
    for Epoch in tqdm(range(1,EPOCHS+1)):
        train_loss=train(model,train_loader,optimizer,log_interval=31)
        #valid_loss,valid_accuracy = evaluate(model, validation_loader)

        print("\n[EPOCH:{}]\t Train Loss:{:.4f}".format(Epoch,train_loss))

        #임베딩 기록
        if Epoch % 10 ==0:
            train_result = []
            train_labels = []
            train_paths = []
            model.eval()
            with torch.no_grad():
                print("Update train result")
                for img,_,_,label,paths in tqdm(train_loader):
                    train_result.append(model(img.to(DEVICE),tsne=False ).cpu().numpy())
                    train_labels.append(label)
                    train_paths.append(paths)
                
                train_result = np.concatenate(train_result)
                train_labels = np.concatenate(train_labels)
                train_paths = np.concatenate(train_paths)
                
                
                # train classifier
                tree = XGBClassifier(seed=random_state,use_label_encoder=False)
                tree.fit(train_result, train_labels)
                pred_probs = tree.predict_proba(train_result)[:,1]
                preds = [ 1 if x > 0.5 else 0 for x in pred_probs]
                accuracy = accuracy_score(train_labels, preds)
                print("\n[EPOCH:{}]\t Classifier Train Accuracy:{:.4f}".format(Epoch,accuracy))


                train_result_list.append(train_result)
                train_label_list.append(train_labels)
                train_path_list.append(train_paths)

            valid_result = []
            valid_labels = []
            valid_paths = []
            with torch.no_grad():
                print("Update valid result")
                for img,label,paths in tqdm(validation_loader):
                    valid_result.append(model(img.to(DEVICE),tsne=False).cpu().numpy())
                    valid_labels.append(label)
                    valid_paths+=paths

            valid_result = np.concatenate(valid_result)
            valid_labels = np.concatenate(valid_labels)
            

            # trained classifier inference
            # train classifier
            pred_probs = tree.predict_proba(valid_result)[:,1]
            preds = [ 1 if x > 0.5 else 0 for x in pred_probs]
            valid_accuracy = accuracy_score(valid_labels, preds)
            
            best_valid_acc = valid_accuracy if valid_accuracy >= best_valid_acc else best_valid_acc
            print("\n[EPOCH:{}]\t Classifier Valid Accuracy:{:.4f}".format(Epoch,valid_accuracy))

            valid_result_list.append(valid_result)
            valid_label_list.append(valid_labels)   
            valid_path_list.append(valid_paths)
            valid_probs_list.append(pred_probs)

            test_result = []
            test_labels = []

            # classifier testset inference

            with torch.no_grad():
                for img,label,paths in tqdm(test_loader):
                    test_result.append(model(img.to(DEVICE),tsne=False).cpu().numpy())
                    test_labels.append(label)
            test_result = np.concatenate(test_result)
            test_labels = np.concatenate(test_labels)

            pred_probs = tree.predict(test_result)
            preds = [ 1 if x > 0.5 else 0 for x in pred_probs]
            test_accuracy = accuracy_score(test_labels, preds)

            #해당 valid가 최대일 때 accuracy 기록
            best_test_acc = test_accuracy if valid_accuracy >= best_valid_acc else best_test_acc
            print("\n[EPOCH:{}]\t Classifier Test Accuracy:{:.4f}".format(Epoch,test_accuracy))
           
            wandb.log({
                "train {}fold loss".format(data_ind) : train_loss,
                "valid {}fold Accuracy".format(data_ind) : valid_accuracy,
                "test {}fold Accuracy".format(data_ind) : test_accuracy},
                commit=True,
                step=Epoch)
    wandb.run.summary.update({"best_valid_{}fold_acc".format(data_ind) : best_valid_acc,
                              "best_test_{}fold_acc".format(data_ind) : best_test_acc,})




