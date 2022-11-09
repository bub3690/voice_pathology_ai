import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import librosa

from .Data import PhraseData


# # 데이터 정의
# - 추가적으로 데이터의 크기를 맞춰주기 위해 3초로 padding 및 truncate 실시 https://sequencedata.tistory.com/25 FixAudioLength
# - 논문에서는 400frame으로 설정.(여기서는 500frame)
# - 전처리 방법 결정.
# 
# 데이터 로더


classes = ["healthy","pathology"]


class svd_dataset(Dataset):
    def __init__(self,
                data_path_list,
                y_label_list,
                classes,
                mfcc_params,
                mel_params,
                spectro_params,
                transform=None,
                normalize=None,
                mfcc_normalize=None):

        #클래스에서 사용할 인자를 받아 인스턴스 변수로 저장하는 일을 한다.
        #예를들면, 이미지의 경로 리스트를 저장하는 일을 하게 된다.
        
        #data_num : k 개 데이터 셋 중 어떤것을 쓸지
        #test인지 아닌지.
        
        self.path_list = data_path_list
        self.label = y_label_list # label data
        self.classes=classes
        self.transform=transform
        self.normalize=normalize
        self.mfcc_normalize = mfcc_normalize

        # sweep params
        self.mel_params = mel_params
        self.spectro_params = spectro_params
        self.mfcc_params = mfcc_params
        #sr,n_mfcc,lifter, hop_length , win_length , n_mels , n_fft , f_max , batch_size
    
    def __len__(self):
        return len(self.path_list)
        #데이터 셋의 길이를 정수로 반환한다.     
    
    def __getitem__(self, idx):
        """
        1. path를 받아서, 소리에서 mfcc를 추출
        2. mfcc를 224프레임으로 패딩.
        3. resnet에 사용되기 위해 3채널로 복사(rgb 처럼)
        4. 0~1 정규화
        
        """
        sig = PhraseData.phrase_dict[ str(self.path_list[idx])+'-phrase.wav'] 
        #sig = preemphasis(sig)
        
        origin_length = sig.shape[0]
        
        if sig.shape[0] > self.mfcc_params["sr"]*2:
            origin_length = self.mfcc_params["sr"]*2
        
        origin_frame_size = 1 + int(np.floor(origin_length//self.mel_params["hop_length"]))
        
        length = self.mfcc_params["sr"]*2 #sample rate *2 padding을 위한 파라미터 (하이퍼 파라미터로인해 사이즈는 계속 바뀐다.)
        pad1d = lambda a, i: a[0:i] if a.shape[0] > i else np.hstack((a, np.zeros((i-a.shape[0]))))        
        sig = pad1d(sig,length)        
        
        ###signal norm
        sig = (sig-sig.mean())/sig.std()
        ###
        
        mel_feature = librosa.feature.melspectrogram(y=sig,
                                                    sr=self.mel_params["sr"],
                                                    # hyp param
                                                    n_mels = self.mel_params["n_mels"],
                                                    n_fft = self.mel_params["n_fft"],
                                                    win_length = self.mel_params["win_length"],
                                                    hop_length = self.mel_params["hop_length"],
                                                    fmax = self.mel_params["f_max"]
                                                    )
        mel_feature = librosa.core.power_to_db(mel_feature,ref=np.max) 
        
        
        if self.transform:
            mel_feature = self.transform(mel_feature).type(torch.float32)# 데이터 0~1 정규화
            MSF = torch.stack([mel_feature, mel_feature, mel_feature])# 3채널로 복사.
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


#데이터 로더 제작 함수
def load_data(
    X_train_list,
    X_valid_list,
    Y_train_list,
    Y_valid_list,
    BATCH_SIZE,
    spectro_run_config,
    mel_run_config,
    mfcc_run_config,
    
    ):

    train_loader = DataLoader(dataset = svd_dataset(
                                                X_train_list,
                                                Y_train_list,
                                                classes,
                                                mfcc_params=mfcc_run_config,
                                                mel_params=mel_run_config,
                                                spectro_params=spectro_run_config,
                                                transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                                #mfcc_normalize=(53.5582, 217.43),
                                            ),
                                            batch_size = BATCH_SIZE,
                                            shuffle = True,
                                            #worker_init_fn=seed_worker
                                            ) # 순서가 암기되는것을 막기위해.

    validation_loader = DataLoader(dataset = 
                                            svd_dataset(
                                                X_valid_list,
                                                Y_valid_list,
                                                classes,
                                                mfcc_params=mfcc_run_config,
                                                mel_params=mel_run_config,
                                                spectro_params=spectro_run_config,
                                                transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                                #mfcc_normalize=(53.5582, 217.43),
                                            ),
                                            batch_size = BATCH_SIZE,
                                            shuffle = True,
                                            #worker_init_fn=seed_worker
                                            )

    return train_loader,validation_loader


def load_test_data(X_test,Y_test,BATCH_SIZE,spectro_run_config,mel_run_config,mfcc_run_config):
    test_loader = DataLoader(dataset = 
                                            svd_dataset(
                                                X_test,
                                                Y_test,
                                                classes,
                                                mfcc_params=mfcc_run_config,
                                                mel_params=mel_run_config,
                                                spectro_params=spectro_run_config,
                                                transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                                #mfcc_normalize=(53.5582, 217.43),
                                            ),
                                            batch_size = BATCH_SIZE,
                                            shuffle = True,
                                            #worker_init_fn=seed_worker
                                            ) # 순서가 암기되는것을 막기위해.        


    return test_loader


