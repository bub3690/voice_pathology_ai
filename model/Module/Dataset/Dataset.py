import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import librosa

from .Data import PhraseData
import torchaudio.transforms as T


###
# augmentation
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(50,78)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

###

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
                is_normalize,
                norm_mean_list,
                norm_std_list,
                augmentation=[],
                augment_params=dict(),
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

        # sweep params
        self.mel_params = mel_params
        self.spectro_params = spectro_params
        self.mfcc_params = mfcc_params
        #sr,n_mfcc,lifter, hop_length , win_length , n_mels , n_fft , f_max , batch_size

        self.is_train = is_train

        #noramlize 관련
        self.is_normalize = is_normalize
        if is_normalize:
            self.normalize=transforms.Normalize((norm_mean_list[1],norm_mean_list[1],norm_mean_list[1]),(norm_std_list[1],norm_std_list[1],norm_std_list[1]))
        else:
            self.normalize=None

        #augmentation들
        self.crop = None
        self.spec_augment = None
        self.augment_params = augment_params
        
        if "crop" in augmentation:
            self.crop = transforms.RandomApply([
                                                Cutout(self.augment_params['crop'][0],
                                                self.augment_params['crop'][1]),
                                                ],
                                                p = self.augment_params['crop'][2])
        if "spec_augment" in augmentation:
            self.spec_augment = transforms.RandomApply([
                                                    transforms.Compose([T.TimeMasking(time_mask_param=self.augment_params['spec_augment'][0]),
                                                                        T.FrequencyMasking(freq_mask_param=self.augment_params['spec_augment'][1]),],)
                                               ],
                                               p=self.augment_params['spec_augment'][2])

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
        
        if sig.shape[0] > self.mel_params["sr"]*2:
            origin_length = self.mel_params["sr"]*2
        
        origin_frame_size = 1 + int(np.floor(origin_length//self.mel_params["hop_length"]))
        
        length = self.mel_params["sr"]*2 #sample rate *2 padding을 위한 파라미터 (하이퍼 파라미터로인해 사이즈는 계속 바뀐다.)
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
                MSF = self.normalize(MSF)
            

            if self.is_train:
                #spec augment
                if self.crop:
                    MSF = self.crop(MSF)
                if self.spec_augment:
                    MSF = self.spec_augment(MSF)

        else:
            pass
            #print("else")
            mel_feature = torch.from_numpy(mel_feature).type(torch.float32)
            mel_feature=mel_feature.unsqueeze(0)#cnn 사용위해서 추가
            #MFCCs = MFCCs.permute(2, 0, 1)
        return MSF, self.classes.index(self.label[idx])

class svd_dataset_harmonic(Dataset):
    def __init__(self,
                data_path_list,
                y_label_list,
                classes,
                mfcc_params,
                mel_params,
                spectro_params,
                is_normalize,
                norm_mean_list,
                norm_std_list,
                is_train=False,
                transform=None,):

        #클래스에서 사용할 인자를 받아 인스턴스 변수로 저장하는 일을 한다.
        #예를들면, 이미지의 경로 리스트를 저장하는 일을 하게 된다.
        
        #data_num : k 개 데이터 셋 중 어떤것을 쓸지
        #test인지 아닌지.
        
        self.path_list = data_path_list
        self.label = y_label_list # label data
        self.classes=classes
        self.transform=transform
        
        

        # sweep params
        self.mel_params = mel_params
        self.spectro_params = spectro_params
        self.mfcc_params = mfcc_params
        #sr,n_mfcc,lifter, hop_length , win_length , n_mels , n_fft , f_max , batch_size

        #noramlize 관련
        self.is_normalize = is_normalize
        if is_normalize:
            self.normalize=transforms.Normalize((norm_mean_list[1],norm_mean_list[1],norm_mean_list[1]),(norm_std_list[1],norm_std_list[1],norm_std_list[1]))
        else:
            self.normalize=None

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
        
        if sig.shape[0] > self.mel_params["sr"]*2:
            origin_length = self.mel_params["sr"]*2
        
        origin_frame_size = 1 + int(np.floor(origin_length//self.mel_params["hop_length"]))
        
        length = self.mel_params["sr"]*2 #sample rate *2 padding을 위한 파라미터 (하이퍼 파라미터로인해 사이즈는 계속 바뀐다.)
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

        mel_feature1 = mel_feature.copy()
        mel_feature2 = mel_feature.copy()
        mel_feature3 = mel_feature.copy()

        mel_feature1[:42,:] = 0
        mel_feature2[42:42*2,:] = 0
        mel_feature3[42*2:42*3,:] = 0

        if self.transform:
            mel_feature1 = self.transform(mel_feature1).type(torch.float32)# 
            mel_feature2 = self.transform(mel_feature2).type(torch.float32)# 
            mel_feature3 = self.transform(mel_feature3).type(torch.float32)# 

            MSF = torch.stack([mel_feature1, mel_feature2, mel_feature3])# 3채널로 복사.
            MSF = MSF.squeeze(dim=1)
            
            # global normalize
            if self.normalize:
                MSF = self.normalize(MSF)
        else:
            pass
            #print("else")
            mel_feature = torch.from_numpy(mel_feature).type(torch.float32)
            mel_feature=mel_feature.unsqueeze(0)#cnn 사용위해서 추가
            #MFCCs = MFCCs.permute(2, 0, 1)
        return MSF, self.classes.index(self.label[idx])



class svd_dataset_msf(Dataset):
    def __init__(self,
                data_path_list,
                y_label_list,
                classes,
                mfcc_params,
                mel_params,
                spectro_params,
                is_normalize,
                norm_mean_list,
                norm_std_list,
                augmentation=[],
                augment_params=dict(),
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
        self.spectro_params = spectro_params
        self.mfcc_params = mfcc_params
        #sr,n_mfcc,lifter, hop_length , win_length , n_mels , n_fft , f_max , batch_size

        #noramlize 관련
        self.is_normalize = is_normalize
        if is_normalize:
            self.normalize=transforms.Normalize((norm_mean_list[0],norm_mean_list[1],norm_mean_list[1]),(norm_std_list[0],norm_std_list[1],norm_std_list[1]))
            self.mfcc_normalize=(norm_mean_list[2],norm_std_list[2])
        else:
            self.normalize=None
            self.mfcc_normalize=None


        #augmentation들
        self.crop = None
        self.spec_augment = None
        self.augment_params = augment_params
        
        if "crop" in augmentation:
            self.crop = transforms.RandomApply([
                                                Cutout(self.augment_params['crop'][0],
                                                self.augment_params['crop'][1]),
                                                ],
                                                p = self.augment_params['crop'][2])
        if "spec_augment" in augmentation:
            self.spec_augment = transforms.RandomApply([
                                                    transforms.Compose([T.TimeMasking(time_mask_param=self.augment_params['spec_augment'][0]),
                                                                        T.FrequencyMasking(freq_mask_param=self.augment_params['spec_augment'][1]),],)
                                               ],
                                               p=self.augment_params['spec_augment'][2])

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
        
        stft = librosa.stft(sig, win_length=self.spectro_params["win_length"],
                                n_fft=self.spectro_params["n_fft"],
                                hop_length=self.spectro_params["hop_length"]
                            )
        magnitude = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(magnitude)
        
        log_spectrogram = log_spectrogram[:self.mel_params["n_mels"],:] 
        
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

        sig_torch=torch.tensor(sig, dtype=torch.float32)
        
        MFCC = T.MFCC(
                        sample_rate = self.mfcc_params["sr"],
                        n_mfcc = self.mfcc_params["n_mfcc"],
                        melkwargs={
                            'n_fft': self.mfcc_params["n_fft"],
                            'n_mels': self.mfcc_params["n_mels"],
                            'hop_length': self.mfcc_params["hop_length"],
                            'mel_scale': self.mfcc_params["mel_scale"],
                            'win_length' : self.mfcc_params["win_length"],
                            'f_max': self.mfcc_params["f_max"]
                        }
                    )
        
        MFCCs=MFCC(sig_torch)
        
        MFCCs = MFCCs[1:,]
        if self.mfcc_normalize:
           MFCCs=(MFCCs-self.mfcc_normalize[0])/self.mfcc_normalize[1]
        (nframes, ncoeff) = MFCCs.shape
        cep_lifter = self.mfcc_params["lifter"]
        
        time_len = mel_feature.shape[1]
 
        
        if cep_lifter > 0:
            n = np.arange(ncoeff)
            lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
            MFCCs *= lift

        
        
        if self.transform:
            log_spectrogram = self.transform(log_spectrogram).type(torch.float32)
            mel_feature = self.transform(mel_feature).type(torch.float32)# 데이터 0~1 정규화
            MSF = torch.stack([log_spectrogram, mel_feature, mel_feature])# 3채널로 복사.
            MSF = MSF.squeeze(dim=1)
            
            MFCCs=MFCCs.type(torch.float32)# 타입 변화
            #MFCCs = self.transform(MFCCs).type(torch.float32)# 데이터 0~1 정규화
            MFCCs=MFCCs.squeeze().mean(axis=1)
            
            #mfcc norm
            #if self.mfcc_normalize:
            #    MFCCs=(MFCCs-self.mfcc_normalize[0])#/self.mfcc_normalize[1]            
            
            # global normalize
            if self.normalize:
                #MFCCs=self.normalize(MFCCs)
                MSF = self.normalize(MSF)
            if self.is_train:
                #spec augment
                if self.crop:
                    MSF = self.crop(MSF)
                if self.spec_augment:
                    MSF = self.spec_augment(MSF)
    
        else:
            pass
            #print("else")
            mel_feature = torch.from_numpy(mel_feature).type(torch.float32)
            mel_feature=mel_feature.unsqueeze(0)#cnn 사용위해서 추가
            #MFCCs = MFCCs.permute(2, 0, 1)
        return MSF,MFCCs, self.classes.index(self.label[idx])        


class svd_dataset_wav(Dataset):
    def __init__(self,
                data_path_list,
                y_label_list,
                classes,
                mfcc_params,
                mel_params,
                spectro_params,
                is_normalize,
                norm_mean_list,
                norm_std_list,
                augmentation=[],
                augment_params=dict(),
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

        # sweep params
        self.mel_params = mel_params
        self.spectro_params = spectro_params
        self.mfcc_params = mfcc_params
        #sr,n_mfcc,lifter, hop_length , win_length , n_mels , n_fft , f_max , batch_size

        self.is_train = is_train

        #noramlize 관련
        self.is_normalize = is_normalize
        if is_normalize:
            self.normalize=transforms.Normalize((norm_mean_list[1],norm_mean_list[1],norm_mean_list[1]),(norm_std_list[1],norm_std_list[1],norm_std_list[1]))
        else:
            self.normalize=None

        #augmentation들
        self.crop = None
        self.spec_augment = None
        self.augment_params = augment_params
        
        if "crop" in augmentation:
            self.crop = transforms.RandomApply([
                                                Cutout(self.augment_params['crop'][0],
                                                self.augment_params['crop'][1]),
                                                ],
                                                p = self.augment_params['crop'][2])
        if "spec_augment" in augmentation:
            self.spec_augment = transforms.RandomApply([
                                                    transforms.Compose([T.TimeMasking(time_mask_param=self.augment_params['spec_augment'][0]),
                                                                        T.FrequencyMasking(freq_mask_param=self.augment_params['spec_augment'][1]),],)
                                               ],
                                               p=self.augment_params['spec_augment'][2])

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
        
        if sig.shape[0] > self.mel_params["sr"]*2:
            origin_length = self.mel_params["sr"]*2
        
        origin_frame_size = 1 + int(np.floor(origin_length//self.mel_params["hop_length"]))
        
        length = self.mel_params["sr"]*2 #sample rate *2 padding을 위한 파라미터 (하이퍼 파라미터로인해 사이즈는 계속 바뀐다.)
        pad1d = lambda a, i: a[0:i] if a.shape[0] > i else np.hstack((a, np.zeros((i-a.shape[0]))))        
        sig = pad1d(sig,length)        
        
        ###signal norm
        sig = (sig-sig.mean())/sig.std()
        ###

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
        sig=torch.from_numpy(sig).type(torch.float32)# 타입 변화
        sig=sig.unsqueeze(0)
        
        return sig, self.classes.index(self.label[idx])        
        
        if self.transform:
            mel_feature = self.transform(mel_feature).type(torch.float32)# 데이터 0~1 정규화
            MSF = torch.stack([mel_feature, mel_feature, mel_feature])# 3채널로 복사.
            MSF = MSF.squeeze(dim=1)    
            
            # global normalize
            if self.normalize:
                MSF = self.normalize(MSF)
            

            if self.is_train:
                #spec augment
                if self.crop:
                    MSF = self.crop(MSF)
                if self.spec_augment:
                    MSF = self.spec_augment(MSF)

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
    is_normalize,
    norm_mean_list,
    norm_std_list,
    model,
    augment,
    augment_params
    ):
    
    if model=='baseline':
        train_loader = DataLoader(dataset = svd_dataset(
                                                    X_train_list,
                                                    Y_train_list,
                                                    classes,
                                                    mfcc_params=mfcc_run_config,
                                                    mel_params=mel_run_config,
                                                    spectro_params=spectro_run_config,
                                                    transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                    is_normalize=is_normalize,
                                                    norm_mean_list=norm_mean_list,
                                                    norm_std_list=norm_std_list,
                                                    is_train=True,
                                                    augmentation=augment,
                                                    augment_params=augment_params,
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
                                                    is_normalize=is_normalize,
                                                    norm_mean_list=norm_mean_list,
                                                    norm_std_list=norm_std_list,
                                                    #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                                    #mfcc_normalize=(53.5582, 217.43),
                                                ),
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                #worker_init_fn=seed_worker
                                                )
    elif model=='msf':
        train_loader = DataLoader(dataset = svd_dataset_msf(
                                                    X_train_list,
                                                    Y_train_list,
                                                    classes,
                                                    mfcc_params=mfcc_run_config,
                                                    mel_params=mel_run_config,
                                                    spectro_params=spectro_run_config,
                                                    transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                    is_normalize=is_normalize,
                                                    norm_mean_list=norm_mean_list,
                                                    norm_std_list=norm_std_list,
                                                    is_train=True,
                                                    augmentation=augment,
                                                    augment_params=augment_params,
                                                    #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                                    #mfcc_normalize=(53.5582, 217.43),
                                                ),
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                #worker_init_fn=seed_worker
                                                ) # 순서가 암기되는것을 막기위해.

        validation_loader = DataLoader(dataset= svd_dataset_msf(
                                                    X_valid_list,
                                                    Y_valid_list,
                                                    classes,
                                                    mfcc_params=mfcc_run_config,
                                                    mel_params=mel_run_config,
                                                    spectro_params=spectro_run_config,
                                                    transform=transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                    is_normalize=is_normalize,
                                                    norm_mean_list=norm_mean_list,
                                                    norm_std_list=norm_std_list,
                                                    #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                                    #mfcc_normalize=(53.5582, 217.43),
                                                ),
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                #worker_init_fn=seed_worker
                                                )
    elif model=='decomp':
        train_loader = DataLoader(dataset = svd_dataset_harmonic(
                                                    X_train_list,
                                                    Y_train_list,
                                                    classes,
                                                    mfcc_params=mfcc_run_config,
                                                    mel_params=mel_run_config,
                                                    spectro_params=spectro_run_config,
                                                    transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                    is_normalize=is_normalize,
                                                    norm_mean_list=norm_mean_list,
                                                    norm_std_list=norm_std_list,
                                                    is_train=True,
                                                    augmentation=augment,
                                                    #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                                    #mfcc_normalize=(53.5582, 217.43),
                                                ),
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                #worker_init_fn=seed_worker
                                                ) # 순서가 암기되는것을 막기위해.
        validation_loader = DataLoader(dataset = 
                                                svd_dataset_harmonic(
                                                    X_valid_list,
                                                    Y_valid_list,
                                                    classes,
                                                    mfcc_params=mfcc_run_config,
                                                    mel_params=mel_run_config,
                                                    spectro_params=spectro_run_config,
                                                    transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                    is_normalize=is_normalize,
                                                    norm_mean_list=norm_mean_list,
                                                    norm_std_list=norm_std_list,
                                                    #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                                    #mfcc_normalize=(53.5582, 217.43),
                                                ),
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                #worker_init_fn=seed_worker
                                                )
    else:
        train_loader = DataLoader(dataset = svd_dataset(
                                                    X_train_list,
                                                    Y_train_list,
                                                    classes,
                                                    mfcc_params=mfcc_run_config,
                                                    mel_params=mel_run_config,
                                                    spectro_params=spectro_run_config,
                                                    transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                    is_normalize=is_normalize,
                                                    norm_mean_list=norm_mean_list,
                                                    norm_std_list=norm_std_list,
                                                    is_train=True,
                                                    augmentation=augment
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
                                                    is_normalize=is_normalize,
                                                    norm_mean_list=norm_mean_list,
                                                    norm_std_list=norm_std_list,
                                                    #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                                    #mfcc_normalize=(53.5582, 217.43),
                                                ),
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                #worker_init_fn=seed_worker
                                                )
    return train_loader,validation_loader


def load_test_data(X_test,Y_test,BATCH_SIZE,spectro_run_config,mel_run_config,mfcc_run_config,is_normalize,norm_mean_list,norm_std_list,model):
    if model=='baseline':
        test_loader = DataLoader(dataset = svd_dataset(
                                            X_test,
                                            Y_test,
                                            classes,
                                            mfcc_params=mfcc_run_config,
                                            mel_params=mel_run_config,
                                            spectro_params=spectro_run_config,
                                            transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                            is_normalize=is_normalize,
                                            norm_mean_list=norm_mean_list,
                                            norm_std_list=norm_std_list,
                                            #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                            #mfcc_normalize=(53.5582, 217.43),
                                        ),
                                        batch_size = BATCH_SIZE,
                                        shuffle = True,
                                        #worker_init_fn=seed_worker
                                        ) # 순서가 암기되는것을 막기위해.
    elif model=='msf':
        test_loader = DataLoader(dataset =  svd_dataset_msf(
                                                    X_test,
                                                    Y_test,
                                                    classes,
                                                    mfcc_params=mfcc_run_config,
                                                    mel_params=mel_run_config,
                                                    spectro_params=spectro_run_config,
                                                    transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                    is_normalize=is_normalize,
                                                    norm_mean_list=norm_mean_list,
                                                    norm_std_list=norm_std_list,
                                                    #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                                    #mfcc_normalize=(53.5582, 217.43),
                                                ),
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                #worker_init_fn=seed_worker
                                                ) # 순서가 암기되는것을 막기위해.
    elif model=='decomp':
        test_loader = DataLoader(dataset =  svd_dataset_harmonic(
                                                    X_test,
                                                    Y_test,
                                                    classes,
                                                    mfcc_params=mfcc_run_config,
                                                    mel_params=mel_run_config,
                                                    spectro_params=spectro_run_config,
                                                    transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                                    is_normalize=is_normalize,
                                                    norm_mean_list=norm_mean_list,
                                                    norm_std_list=norm_std_list,
                                                    #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                                    #mfcc_normalize=(53.5582, 217.43),
                                                ),
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                #worker_init_fn=seed_worker
                                                ) # 순서가 암기되는것을 막기위해.
    else:
        test_loader = DataLoader(dataset = svd_dataset(
                                            X_test,
                                            Y_test,
                                            classes,
                                            mfcc_params=mfcc_run_config,
                                            mel_params=mel_run_config,
                                            spectro_params=spectro_run_config,
                                            transform = transforms.ToTensor(),#이걸 composed로 고쳐서 전처리 하도록 수정.
                                            is_normalize=is_normalize,
                                            norm_mean_list=norm_mean_list,
                                            norm_std_list=norm_std_list,
                                            #normalize=transforms.Normalize((-11.4805,-54.7723,-54.7723),(16.87,19.0226,19.0226)),
                                            #mfcc_normalize=(53.5582, 217.43),
                                        ),
                                        batch_size = BATCH_SIZE,
                                        shuffle = True,
                                        #worker_init_fn=seed_worker
                                        ) # 순서가 암기되는것을 막기위해.        
    
    
    return test_loader





