import torch
import torch.nn as nn # 인공 신경망 모델들 모아놓은 모듈
import torch.nn.functional as F #그중 자주 쓰이는것들을 F로
import torchvision.models as models
import torchaudio
import torchaudio.transforms as T
import librosa

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
        #print(x.size())
        x = self.model(x)
        #x  = self.fc(x)
        return x


class Resnet_wav(nn.Module):
    def __init__(self, mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        super(Resnet_wav, self).__init__()
        # if "center=True" of stft, padding = win_len / 2

        #self.num_ftrs = 63

        self.res = ResLayer()


        # self.spec = T.Spectrogram(n_fft=win_len,hop_length=hop_len,power=2)

        # self.mel_scale = T.MelScale(
        #     n_mels=mel_bins, sample_rate=16000, n_stft=win_len // 2 + 1
        #     )

        self.power_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        self.mel_scale = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            win_length=win_len,
            hop_length=hop_len,
            n_mels=mel_bins,
            f_min=0,
            f_max=8000,
            center=True,
            pad_mode="constant",
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            window_fn=torch.hann_window
        )

        # stretch_factor=0.8
        # self.spec_aug = torch.nn.Sequential(
        #     T.TimeStretch(stretch_factor, fixed_rate=True),
        #     T.FrequencyMasking(freq_mask_param=80),
        #     T.TimeMasking(time_mask_param=40),
        # )        

    def forward(self, x):
        #spec = self.spec(x)
        #mel = self.mel_spectrogram(x)
        mel = self.mel_scale(x)
        mel = torchaudio.functional.amplitude_to_DB(mel,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel)) )
        
        #mel = self.power_to_db(mel)
        #mel = self.spec_aug(mel)
        mel = torch.squeeze(mel,dim=1)
        #mel = (mel-torch.mean(mel))/torch.std(mel)
        #out = out.mean(axis=2)
        #out=self.fc(out)
        
        #concated_feature = torch.concat([mel,out],axis=2)

        out = torch.stack([mel,mel,mel],axis=1)
        #print(out.size())
        out=self.res(out)
        return out
 
class Attention_wav(nn.Module):
    def __init__(self, mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        super(Attention_wav, self).__init__()
        # if "center=True" of stft, padding = win_len / 2

        #self.num_ftrs = 63

        self.res = ResLayer()


        # self.spec = T.Spectrogram(n_fft=win_len,hop_length=hop_len,power=2)

        # self.mel_scale = T.MelScale(
        #     n_mels=mel_bins, sample_rate=16000, n_stft=win_len // 2 + 1
        #     )

        self.power_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        self.mel_scale = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            win_length=win_len,
            hop_length=hop_len,
            n_mels=mel_bins,
            f_min=0,
            f_max=8000,
            center=True,
            pad_mode="constant",
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            window_fn=torch.hann_window
        )

        # stretch_factor=0.8
        # self.spec_aug = torch.nn.Sequential(
        #     T.TimeStretch(stretch_factor, fixed_rate=True),
        #     T.FrequencyMasking(freq_mask_param=80),
        #     T.TimeMasking(time_mask_param=40),
        # )        

    def forward(self, x):
        #spec = self.spec(x)
        #mel = self.mel_spectrogram(x)
        mel = self.mel_scale(x)
        mel = torchaudio.functional.amplitude_to_DB(mel,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel)) )
        
        #mel = self.power_to_db(mel)
        #mel = self.spec_aug(mel)
        mel = torch.squeeze(mel,dim=1)
        #mel = (mel-torch.mean(mel))/torch.std(mel)
        #out = out.mean(axis=2)
        #out=self.fc(out)
        
        #concated_feature = torch.concat([mel,out],axis=2)

        out = torch.stack([mel,mel,mel],axis=1)
        #print(out.size())
        out=self.res(out)
        return out

class Resnet_wav_latefusion(nn.Module):
    def __init__(self, mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        super(Resnet_wav_latefusion, self).__init__()

        self.res_h = models.resnet18(pretrained=True).cuda()
        self.res_l = models.resnet18(pretrained=True).cuda()
        self.res_n = models.resnet18(pretrained=True).cuda()
        #self.num_ftrs = self.model.fc.out_features

        self.num_ftrs = self.res_h.fc.out_features
        
        self.pad1d = lambda a, i: a[0:i] if a.shape[0] > i else torch.hstack((a, torch.zeros((i-a.shape[0]))))   

        self.mel_scale = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            win_length=win_len,
            hop_length=hop_len,
            n_mels=mel_bins,
            f_min=0,
            f_max=8000,
            center=True,
            pad_mode="constant",
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            window_fn=torch.hann_window
        )

        self.fc = nn.Sequential(       
                            nn.Linear(self.num_ftrs*3, self.num_ftrs),
                            nn.BatchNorm1d(self.num_ftrs),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(self.num_ftrs,128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(128,64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,2),
                            )


    def forward(self, x_list):
        # h, l ,n 순으로 입력
        mel_h = self.mel_scale(x_list[:,0,...])
        mel_h = torchaudio.functional.amplitude_to_DB(mel_h,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel_h)) )
        mel_h = torch.squeeze(mel_h,dim=1)

        mel_l = self.mel_scale(x_list[:,1,...])
        mel_l = torchaudio.functional.amplitude_to_DB(mel_l,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel_l)) )
        mel_l = torch.squeeze(mel_l,dim=1)

        mel_n = self.mel_scale(x_list[:,2,...])
        mel_n = torchaudio.functional.amplitude_to_DB(mel_n,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel_n)) )
        mel_n = torch.squeeze(mel_n,dim=1)

        out_h = torch.stack([mel_h,mel_h,mel_h],axis=1)
        out_l = torch.stack([mel_l,mel_l,mel_l],axis=1)
        out_n = torch.stack([mel_n,mel_n,mel_n],axis=1)

        #print(out.size())
        out_h=self.res_h(out_h)
        out_l=self.res_l(out_l)
        out_n=self.res_n(out_n)

        out = torch.cat([out_h,out_l,out_n],axis=1)
        out = self.fc(out)

        return out


class Resnet_wav_latefusion_all(nn.Module):
    def __init__(self, mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        super(Resnet_wav_latefusion_all, self).__init__()

        self.res_phrase = models.resnet18(pretrained=True).cuda()
        self.res_a = models.resnet18(pretrained=True).cuda()
        self.res_i = models.resnet18(pretrained=True).cuda()
        self.res_u = models.resnet18(pretrained=True).cuda()
        #self.num_ftrs = self.model.fc.out_features

        self.num_ftrs = self.res_phrase.fc.out_features

        self.pad1d = lambda a, i: a[0:i] if a.shape[0] > i else torch.hstack((a, torch.zeros((i-a.shape[0]))))   

        self.mel_scale = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            win_length=win_len,
            hop_length=hop_len,
            n_mels=mel_bins,
            f_min=0,
            f_max=8000,
            center=True,
            pad_mode="constant",
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            window_fn=torch.hann_window
        )

        self.fc = nn.Sequential(       
                            nn.Linear(self.num_ftrs*4, self.num_ftrs),
                            nn.BatchNorm1d(self.num_ftrs),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(self.num_ftrs,128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(128,64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,2),
                            )


    def forward(self, x_list):
        # phrase, a, i ,u 순으로 입력
        mel_phrase = self.mel_scale(x_list[:,0,...])
        mel_phrase = torchaudio.functional.amplitude_to_DB(mel_phrase,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel_phrase)) )
        mel_phrase = torch.squeeze(mel_phrase,dim=1)

        mel_a = self.mel_scale(x_list[:,1,...])
        mel_a = torchaudio.functional.amplitude_to_DB(mel_a,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel_a)) )
        mel_a = torch.squeeze(mel_a,dim=1)

        mel_i = self.mel_scale(x_list[:,2,...])
        mel_i = torchaudio.functional.amplitude_to_DB(mel_i,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel_i)) )
        mel_i = torch.squeeze(mel_i,dim=1)

        mel_u = self.mel_scale(x_list[:,3,...])
        mel_u = torchaudio.functional.amplitude_to_DB(mel_u,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel_u)) )
        mel_u = torch.squeeze(mel_u,dim=1)

        out_phrase = torch.stack([mel_phrase,mel_phrase,mel_phrase],axis=1)
        out_a = torch.stack([mel_a,mel_a,mel_a],axis=1)
        out_i = torch.stack([mel_i,mel_i,mel_i],axis=1)
        out_u = torch.stack([mel_u,mel_u,mel_u],axis=1)

        #print(out.size())
        out_phrase=self.res_phrase(out_phrase)
        out_a=self.res_a(out_a)
        out_i=self.res_i(out_i)
        out_u=self.res_u(out_u)

        out = torch.cat([out_phrase,out_a,out_i,out_u],axis=1)
        out = self.fc(out)

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
    elif model_name=='wav_res':
        model = Resnet_wav(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()      
    elif model_name=='wav_res_latefusion':
        model = Resnet_wav_latefusion(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
    elif model_name=='wav_res_concat':
        model = Attention_wav(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
        #model = Resnet_wav(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
    elif model_name=='wav_res_concat_fusion': # vowel들 concat + latefusion
        model = Resnet_wav_latefusion(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
    elif model_name == 'wav_res_concat_allfusion':
        model = Resnet_wav_latefusion_all(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
    elif model_name == 'baseline':
        model = ResLayer().cuda()
    elif model_name == 'decomp':
        model = ResLayer().cuda()
    elif model_name == 'se_resnet18':
        model = se_resnet18().cuda()

    return model