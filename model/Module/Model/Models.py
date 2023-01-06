import torch
import torch.nn as nn # 인공 신경망 모델들 모아놓은 모듈
import torch.nn.functional as F #그중 자주 쓰이는것들을 F로
import torchvision.models as models
import torchaudio
import torchaudio.transforms as T

from torchvision.models import ResNet
import timm


# # RESNET
# 모델
# pretrained
class ResLayer(nn.Module):
    def __init__(self):
        super(ResLayer, self).__init__()
        self.model = models.resnet18(pretrained=True).cuda() 
        #self.num_ftrs = self.model.fc.out_features
        self.num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Sequential(       
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
        # self.fc = nn.Sequential(       
        #     nn.Linear(self.num_ftrs, 64),
        #                     nn.BatchNorm1d(64),
        #                     nn.ReLU(),
        #                     nn.Dropout(p=0.5),
        #                     nn.Linear(64,50),
        #                     nn.BatchNorm1d(50),
        #                     nn.ReLU(),
        #                     nn.Dropout(p=0.5),
        #                     nn.Linear(50,2)
        #                     )
    def forward(self, x):
        x = self.model(x)
        #x  = self.fc(x)
        return x


class Resnet_wav(nn.Module):
    def __init__(self, mel_bins=128, win_len=1024, hop_len=512):
        super(Resnet_wav, self).__init__()
        # if "center=True" of stft, padding = win_len / 2

        #self.num_ftrs = 63

        self.res = ResLayer()
        self.fc = nn.Sequential(
                            nn.Linear(mel_bins, 64),
                             nn.BatchNorm1d(64),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(64,50),
                             nn.BatchNorm1d(50),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(50,2)
                            )

        self.spec = T.Spectrogram(n_fft=win_len,hop_length=hop_len,power=2)

        self.mel_scale = T.MelScale(
            n_mels=mel_bins, sample_rate=16000, n_stft=win_len // 2 + 1
            )

        # stretch_factor=0.8
        # self.spec_aug = torch.nn.Sequential(
        #     T.TimeStretch(stretch_factor, fixed_rate=True),
        #     T.FrequencyMasking(freq_mask_param=80),
        #     T.TimeMasking(time_mask_param=40),
        # )        

    def forward(self, x):
        spec = self.spec(x)
        mel = self.mel_scale(spec)
        #mel = self.spec_aug(mel)
        mel = torch.squeeze(mel,dim=1)
        #out = out.mean(axis=2)
        #out=self.fc(out)
        
        #concated_feature = torch.concat([mel,out],axis=2)

        out = torch.stack([mel,mel,mel],axis=1)
        #print(out.size())
        out=self.res(out)
        return out



class MSF(nn.Module):
    def __init__(self,n_mfcc):
        super(MSF, self).__init__()
        self.n_mfcc = n_mfcc-1
        self.model = models.resnet18(pretrained=True).cuda() 
        self.num_ftrs = self.model.fc.out_features
        
        
        #self.mfcc_fc=nn.Sequential(
        #    nn.Linear(self.n_mfcc, self.n_mfcc),
        #    nn.BatchNorm1d(self.n_mfcc),
        #    nn.ReLU()
        #)
        
        self.spectro_fc=nn.Sequential(
            nn.Linear(self.num_ftrs,self.num_ftrs),
            nn.BatchNorm1d(self.num_ftrs),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            #nn.BatchNorm1d(self.num_ftrs+self.n_mfcc),                
            nn.Linear(self.num_ftrs+self.n_mfcc, 64),
                             nn.BatchNorm1d(64),
                             #nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(64,50),
                             nn.BatchNorm1d(50),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(50,2)
                            )
        

    def forward(self, x, mfccs):
        #mfccs = self.mfcc_fc(mfccs)
        
        x = self.model(x)
        x = self.spectro_fc(x)
        
        x = torch.cat([x,mfccs],axis=1)
        x  = self.fc(x)
        return x



###
#seresnet

def se_resnet18(num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = timm.create_model('legacy_seresnet18',num_classes=2,pretrained=True)
    return model



#####








def model_initialize(model_name,spectro_run_config, mel_run_config, mfcc_run_config):
    if model_name=='msf':
        model = MSF(mfcc_run_config['n_mfcc']).cuda()
    elif model=='wav_phrase':
        model = Resnet_wav(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],hop_len=mel_run_config['hop_length']).cuda()      
    elif model_name == 'baseline':
        model = ResLayer().cuda()
    elif model_name == 'decomp':
        model = ResLayer().cuda()
    elif model_name == 'se_resnet18':
        model = se_resnet18().cuda()

    return model