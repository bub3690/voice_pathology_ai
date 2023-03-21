import torch
import torch.nn as nn # 인공 신경망 모델들 모아놓은 모듈
import torch.nn.functional as F #그중 자주 쓰이는것들을 F로
import torchvision.models as models
import torchvision.transforms
import torchaudio
import torchaudio.transforms as T
import librosa

from torchvision.models import ResNet
import timm

from torchvision.models.resnet import BasicBlock, ResNet
from dropblock import DropBlock2D, LinearScheduler

import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock

from .Ablations import se_resnet18,xception,\
    efficinet_bo




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

    def forward(self, x):
        #print(x.size())
        x = self.model(x)
        #x  = self.fc(x)
        return x


class ResLayer34(nn.Module):
    def __init__(self):
        super(ResLayer34, self).__init__()
        self.model = models.resnet34(pretrained=True).cuda() 
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
        #mel_bins=128,win_len=1024,n_fft=1024, hop_len=512
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

        # self.mel_scale = T.MelSpectrogram(
        #     sample_rate=16000,
        #     n_fft=n_fft,
        #     win_length=win_len,
        #     hop_length=hop_len,
        #     n_mels=mel_bins,
        #     f_min=0,
        #     f_max=8000,
        #     power=1.0,
        #     wkwargs={"periodic":False},
        #     window_fn=torch.hann_window
        # )

        stretch_factor=0.8
        self.spec_aug = torch.nn.Sequential(
            #T.TimeStretch(stretch_factor, fixed_rate=True),
            T.FrequencyMasking(freq_mask_param=80),
            T.TimeMasking(time_mask_param=40),
        )

    #@classmethod
    def sample_min_max(batch):
        batch_size,height,width = batch.size(0),batch.size(1),batch.size(2)
        batch = batch.contiguous().view(batch.size(0), -1)
        batch -= batch.min(1, keepdim=True)[0]
        batch /= batch.max(1, keepdim=True)[0]
        batch = batch.view(batch_size, height, width)
        return batch

    def batch_min_max(batch):
        batch = (batch-batch.min())/(batch.max()-batch.min())
        return batch
    
    def take_log(feature):
        amp2db = torchaudio.transforms.AmplitudeToDB(stype="amplitude")
        amp2db.amin=1e-5
        return amp2db(feature).clamp(min=-50,max=80)
    

    def forward(self, x,augment=False):
        #spec = self.spec(x)
        #mel = self.mel_spectrogram(x)
        mel = self.mel_scale(x)
        
        mel = torchaudio.functional.amplitude_to_DB(mel,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel)) )
        mel = torch.squeeze(mel,dim=1)        
        #mel = (mel-mel.min())/(mel.max()-mel.min())
        #mel = Resnet_wav.take_log(mel)
        
        mel = Resnet_wav.batch_min_max(mel)

        #mel = self.power_to_db(mel)
        #mel = self.spec_aug(mel)
        #mel = (mel-torch.mean(mel))/torch.std(mel)
        #out = out.mean(axis=2)
        #out=self.fc(out)
        
        #concated_feature = torch.concat([mel,out],axis=2)

        out = torch.stack([mel,mel,mel],axis=1)
        #print(out.size())
        out=self.res(out)
        return out

class Resnet_wav_smile(nn.Module):
    def __init__(self, mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        #mel_bins=128,win_len=1024,n_fft=1024, hop_len=512
        super(Resnet_wav_smile, self).__init__()
        # if "center=True" of stft, padding = win_len / 2

        #self.num_ftrs = 63

        self.res = models.resnet18(pretrained=True).cuda() 
        #self.num_ftrs = self.model.fc.out_features
        self.num_ftrs = self.res.fc.in_features

        self.smile_fc = nn.Sequential(       
                            nn.Linear(6373, 2048),
                            nn.BatchNorm1d(2048),
                            nn.ReLU(),
                            nn.Linear(2048, 512)
                            )

        self.res.fc = nn.Sequential(       
            nn.Linear(self.num_ftrs, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Dropout(p=0.5)                         
                            )
        
        self.concated_fc= nn.Sequential(
            nn.Linear(512*2,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64,2)
        )


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

        # self.mel_scale = T.MelSpectrogram(
        #     sample_rate=16000,
        #     n_fft=n_fft,
        #     win_length=win_len,
        #     hop_length=hop_len,
        #     n_mels=mel_bins,
        #     f_min=0,
        #     f_max=8000,
        #     power=1.0,
        #     wkwargs={"periodic":False},
        #     window_fn=torch.hann_window
        # )

        stretch_factor=0.8
        self.spec_aug = torch.nn.Sequential(
            #T.TimeStretch(stretch_factor, fixed_rate=True),
            T.FrequencyMasking(freq_mask_param=80),
            T.TimeMasking(time_mask_param=40),
        )

    #@classmethod
    def sample_min_max(batch):
        batch_size,height,width = batch.size(0),batch.size(1),batch.size(2)
        batch = batch.contiguous().view(batch.size(0), -1)
        batch -= batch.min(1, keepdim=True)[0]
        batch /= batch.max(1, keepdim=True)[0]
        batch = batch.view(batch_size, height, width)
        return batch

    def batch_min_max(batch):
        batch = (batch-batch.min())/(batch.max()-batch.min())
        return batch
    
    def take_log(feature):
        amp2db = torchaudio.transforms.AmplitudeToDB(stype="amplitude")
        amp2db.amin=1e-5
        return amp2db(feature).clamp(min=-50,max=80)
    

    def forward(self, x,handcrafted,augment=False):
        #spec = self.spec(x)
        #mel = self.mel_spectrogram(x)

        mel = self.mel_scale(x)
        
        mel = torchaudio.functional.amplitude_to_DB(mel,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel)) )
        mel = torch.squeeze(mel,dim=1)        
        #mel = (mel-mel.min())/(mel.max()-mel.min())
        #mel = Resnet_wav.take_log(mel)
        
        mel = Resnet_wav.batch_min_max(mel)
        

        #mel = self.power_to_db(mel)
        #mel = self.spec_aug(mel)
        #mel = (mel-torch.mean(mel))/torch.std(mel)
        #out = out.mean(axis=2)
        #out=self.fc(out)
        
        #concated_feature = torch.concat([mel,out],axis=2)

        out = torch.stack([mel,mel,mel],axis=1)
        #print(out.size())
        out=self.res(out)

        smile_out = self.smile_fc(handcrafted)

        out = torch.concat([out,smile_out],axis=1)
        out = self.concated_fc(out)
        return out


class Resnet18_custom(ResNet):
    def __init__(self,):
        super(Resnet18_custom, self).__init__(BasicBlock, [2, 2, 2, 2])


        self.num_ftrs = self.fc.in_features+20

        self.fc = nn.Sequential(       
            nn.Linear(1024, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,2)
                            )
        # self.fc = nn.Sequential(nn.Linear(20,2))

        input_size=512
        hidden_size=512 # time size
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size=hidden_size,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = True)
        self.linear1 = nn.Linear(2048,512)
            


    def forward(self, x):
        # change forward here
        #print(x.size())
        x = self.conv1(x) # 64, 64, 151
        #print('conv1 : ',x.size())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print('maxpool : ',x.size()) # 64, 32, 76
        x = self.layer1(x) # batch, 64, 32, 76




        x = self.layer2(x) # 128, 16, 38

        x = self.layer3(x) # 256, 8, 19

        x = self.layer4(x) # 512, 4, 10
        #print('layer 4 : ',x.size()) #

        #여기서 lstm 투입
        #Channel 1d cnn -> lstm(feature 32)
        batch_size = x.size(0)
        T = x.size(3)
        temporal_feature = x.view(batch_size,-1,T) # [batch_size, T==width, num_features==channels*height] bs,512,4,10 -> bs,10,512*4
        temporal_feature = torch.transpose(temporal_feature,1,2)
        temporal_feature = self.linear1(temporal_feature)        

        _,(temporal_feature,cell_state) = self.lstm(temporal_feature)
        temporal_feature = torch.cat([temporal_feature[-1], temporal_feature[-2]], dim=-1)#batch,20

        
        #x = self.avgpool(x)

        #print(x.size())
        
        #x = torch.flatten(x, 1)
        #x = torch.cat([x,temporal_feature],dim=-1)        
        #x = self.fc(x)
        x = self.fc(temporal_feature)
        return x


class Resnet_wav_temporal(nn.Module):
    def __init__(self, mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        super(Resnet_wav_temporal, self).__init__()
        # if "center=True" of stft, padding = win_len / 2

        #self.num_ftrs = 63

        self.res = Resnet18_custom()


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

        stretch_factor=0.8
        self.spec_aug = torch.nn.Sequential(
            #T.TimeStretch(stretch_factor, fixed_rate=True),
            T.FrequencyMasking(freq_mask_param=80),
            T.TimeMasking(time_mask_param=40),
        )

    #@classmethod
    def sample_min_max(batch):
        batch_size,height,width = batch.size(0),batch.size(1),batch.size(2)
        batch = batch.contiguous().view(batch.size(0), -1)
        batch -= batch.min(1, keepdim=True)[0]
        batch /= batch.max(1, keepdim=True)[0]
        batch = batch.view(batch_size, height, width)
        return batch

    def batch_min_max(batch):
        batch = (batch-batch.min())/(batch.max()-batch.min())
        return batch
    

    def forward(self, x,augment=False):
        #spec = self.spec(x)
        #mel = self.mel_spectrogram(x)
        mel = self.mel_scale(x)
        
        mel = torchaudio.functional.amplitude_to_DB(mel,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel)) )
        mel = torch.squeeze(mel,dim=1)        
        #mel = (mel-mel.min())/(mel.max()-mel.min())
        mel = Resnet_wav_temporal.batch_min_max(mel)

        #mel = self.power_to_db(mel)
        #mel = self.spec_aug(mel)
        #mel = (mel-torch.mean(mel))/torch.std(mel)
        #out = out.mean(axis=2)
        #out=self.fc(out)
        
        #concated_feature = torch.concat([mel,out],axis=2)

        out = torch.stack([mel,mel,mel],axis=1)
        #print(out.size())
        out=self.res(out)
        return out


# 0211 dynamic attention

class DynamicResnet18(ResNet):
    def __init__(self,):
        super(DynamicResnet18, self).__init__(BasicBlock, [2, 2, 2, 2])


    def get_time_attention_layer(self,basis_kernel=1):
        #self.layer1_attention = nn.Sequential()
        
        
        self.layer1_time_avg=nn.AdaptiveAvgPool2d(output_size=(32,1))
        self.layer1_conv1 = nn.Conv1d(64,16,kernel_size=(1,1))
        self.layer1_batch_norm = nn.BatchNorm2d(16)
        self.layer1_score_relu = nn.ReLU()
        self.layer1_conv2 = nn.Conv1d(16,basis_kernel,kernel_size=(1,1))

        torch.nn.init.kaiming_normal_(self.layer1_conv1.weight)
        torch.nn.init.kaiming_normal_(self.layer1_conv2.weight)

        self.layer2_time_avg=nn.AdaptiveAvgPool2d(output_size=(16,1))
        self.layer2_conv1 = nn.Conv1d(128,32,kernel_size=(1,1))
        self.layer2_batch_norm = nn.BatchNorm2d(32)
        self.layer2_score_relu = nn.ReLU()
        self.layer2_conv2 = nn.Conv1d(32,basis_kernel,kernel_size=(1,1))

        torch.nn.init.kaiming_normal_(self.layer2_conv1.weight)
        torch.nn.init.kaiming_normal_(self.layer2_conv2.weight)


        self.layer3_time_avg=nn.AdaptiveAvgPool2d(output_size=(16,1))
        self.layer3_conv1 = nn.Conv1d(256,64,kernel_size=(1,1))
        self.layer3_batch_norm = nn.BatchNorm2d(64)
        self.layer3_score_relu = nn.ReLU()
        self.layer3_conv2 = nn.Conv1d(64,basis_kernel,kernel_size=(1,1))

        torch.nn.init.kaiming_normal_(self.layer3_conv1.weight)
        torch.nn.init.kaiming_normal_(self.layer3_conv2.weight)

            

    def get_freq_attention_layer(self,basis_kernel=4):
        #self.layer1_attention = nn.Sequential()
        
        
        self.layer1_freq_conv1 = nn.Conv1d(64,16,kernel_size=(1,1))
        self.layer1_freq_batch_norm = nn.BatchNorm2d(16)
        self.layer1_freq_score_relu = nn.ReLU()
        self.layer1_freq_conv2 = nn.Conv1d(16,basis_kernel,kernel_size=(1,1))

        torch.nn.init.kaiming_normal_(self.layer1_freq_conv1.weight)
        torch.nn.init.kaiming_normal_(self.layer1_freq_conv2.weight)
        self.layer1_weight = nn.Parameter(torch.randn(basis_kernel, 64, 64, 3, 3),
                                                requires_grad=True)


        self.layer2_freq_conv1 = nn.Conv1d(128,32,kernel_size=(1,1))
        self.layer2_freq_batch_norm = nn.BatchNorm2d(32)
        self.layer2_freq_score_relu = nn.ReLU()
        self.layer2_freq_conv2 = nn.Conv1d(32,basis_kernel,kernel_size=(1,1))

        torch.nn.init.kaiming_normal_(self.layer2_freq_conv1.weight)
        torch.nn.init.kaiming_normal_(self.layer2_freq_conv2.weight)
        self.layer2_weight = nn.Parameter(torch.randn(basis_kernel, 128, 128, 3, 3),
                                                requires_grad=True)        

        self.layer3_freq_conv1 = nn.Conv1d(256,64,kernel_size=(1,1))
        self.layer3_freq_batch_norm = nn.BatchNorm2d(64)
        self.layer3_freq_score_relu = nn.ReLU()
        self.layer3_freq_conv2 = nn.Conv1d(64,basis_kernel,kernel_size=(1,1))

        torch.nn.init.kaiming_normal_(self.layer3_freq_conv1.weight)
        torch.nn.init.kaiming_normal_(self.layer3_freq_conv2.weight)
        self.layer3_weight = nn.Parameter(torch.randn(basis_kernel, 256, 256, 3, 3),
                                                requires_grad=True)


    def get_layer1_freq_attention(self,x):
        temperature = 4
        #score = self.layer1_time_avg(x)
        score = x.mean(dim=3,keepdim=True)
        score = self.layer1_freq_conv1(score)
        #score = score.view(-1,32)
        score = self.layer1_freq_batch_norm(score)
        score = self.layer1_freq_score_relu(score)
        score = self.layer1_freq_conv2(score)
        #score = score.view(-1,1,32,1)
        score = torch.softmax(score/temperature,dim=2)
        return score

    def get_layer2_freq_attention(self,x):
        temperature = 4
        #score = self.layer1_time_avg(x)
        score = x.mean(dim=3,keepdim=True)
        score = self.layer2_freq_conv1(score)
        #score = score.view(-1,32)
        score = self.layer2_freq_batch_norm(score)
        score = self.layer2_freq_score_relu(score)
        score = self.layer2_freq_conv2(score)
        #score = score.view(-1,1,32,1)
        score = torch.softmax(score/temperature,dim=2)
        return score

    def get_layer3_freq_attention(self,x):
        temperature = 1
        #score = self.layer1_time_avg(x)
        score = x.mean(dim=3,keepdim=True)
        score = self.layer3_freq_conv1(score)
        #score = score.view(-1,32)
        score = self.layer3_freq_batch_norm(score)
        score = self.layer3_freq_score_relu(score)
        score = self.layer3_freq_conv2(score)
        #score = score.view(-1,1,32,1)
        score = torch.softmax(score/temperature,dim=2)
        return score



    def get_layer1_attention(self,x):
        temperature = 32
        #score = self.layer1_time_avg(x)
        score = x.mean(dim=2,keepdim=True)# batch,64,1,76
        score = self.layer1_conv1(score)
        #score = score.view(-1,32)
        score = self.layer1_batch_norm(score)
        score = self.layer1_score_relu(score)
        score = self.layer1_conv2(score)
        #score = score.view(-1,1,32,1)
        score = torch.softmax(score/temperature,dim=2)
        #print(score)
        return score#.unsqueeze(2)


    def get_layer2_attention(self,x):
        temperature = 32
        #score = self.layer2_time_avg(x)
        score = x.mean(dim=2,keepdim=True)
        score = self.layer2_conv1(score)
        #score = score.view(-1,32)
        score = self.layer2_batch_norm(score)
        score = self.layer2_score_relu(score)
        score = self.layer2_conv2(score)
        #score = score.view(-1,1,32,1)
        score = torch.softmax(score/temperature,dim=2)#bs,ker,freq,1
        return score#.unsqueeze(2)
    
    def get_layer3_attention(self,x):
        temperature = 32
        #score = self.layer2_time_avg(x)
        score = x.mean(dim=2,keepdim=True)
        score = self.layer3_conv1(score)
        #score = score.view(-1,32)
        score = self.layer3_batch_norm(score)
        score = self.layer3_score_relu(score)
        score = self.layer3_conv2(score)
        #score = score.view(-1,1,32,1)
        score = torch.softmax(score/temperature,dim=2)
        return score#.unsqueeze(2)
            

    def forward(self, x):
        # change forward here
        #print(x.size())
        x = self.conv1(x) # 64, 64, 151
        #print('conv1 : ',x.size())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print('maxpool : ',x.size()) # 64, 32, 76
        x = self.layer1(x) # batch, 64, 32, 76
        batch_size = x.size(0)
        score = self.get_layer1_freq_attention(x)# bs , n_ker,n_freq,1
        score = score.unsqueeze(2)
        #score_time = self.get_layer1_attention(x)
        aggregate_weight = self.layer1_weight.view(-1, 64, 3, 3)
        #print(aggregate_weight.size())
        x = F.conv2d(x, weight=aggregate_weight, bias=None, stride=1, padding=1)
        #print(x.size())
        x = x.view(batch_size, 4, 64, x.size(-2), x.size(-1))# batch, n_ker,n_channel,n_freq,n_frame
        #print('score :',score.size(),'x : ',x.size())

        #print(score)
        x = torch.sum(x * score, dim=1)
        #x = x.sum(score)
        #x = x.mul(score_time)


        #print('layer 1 : ',x.size())
        x = self.layer2(x) # 128, 16, 38
        score = self.get_layer2_freq_attention(x)
        score = score.unsqueeze(2)
        #score_time = self.get_layer1_attention(x)
        aggregate_weight = self.layer2_weight.view(-1, 128, 3, 3)
        #print(aggregate_weight.size())
        x = F.conv2d(x, weight=aggregate_weight, bias=None, stride=1, padding=1)
        #print(x.size())
        x = x.view(batch_size, 4, 128, x.size(-2), x.size(-1))# batch, n_ker,n_channel,n_freq,n_frame
        #print('score :',score.size(),'x : ',x.size())

        #print(score)
        x = torch.sum(x * score, dim=1)
        #score_time = self.get_layer2_attention(x)        
        #print(score.size())
        #x = x.sum(score)
        #x = x.mul(score_time)        

        #print('layer 2 : ',x.size())
        x = self.layer3(x) # 256, 8, 19
        #score = self.get_layer3_freq_attention(x)
        #score_time = self.get_layer3_attention(x)

        #x = x.mul(score)
        #x = x.mul(score_time)     

        #print('layer 3 : ',x.size())
        x = self.layer4(x) # 512, 4, 10
        #print('layer 4 : ',x.size()) #
        

        #print(x.size())
        #score = self.get_atteniton(x)
        #print("score : ",score.size())
        #x = x.mul(score)
        #print('attention : ',x.size())
        x = self.avgpool(x)
        #print(x.size())
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResLayer_attention(nn.Module):
    def __init__(self,tsne=False):
        super(ResLayer_attention, self).__init__()
        self.tsne = tsne
        #self.model = models.resnet18(pretrained=True).cuda()
        self.model = DynamicResnet18()
        # if you need pretrained weights
        self.model.load_state_dict(models.resnet18(pretrained=True).state_dict())

        self.model.get_time_attention_layer()
        self.model.get_freq_attention_layer()
        
        #removed = list(self.model.layer1.children())[:-1]
        #removed[0].add_module("droblock2d",DropBlock2D(block_size=128, drop_prob=1.0).cuda())
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
        
        
        #self.model.fc = nn.Linear(256, 1000)
        self.num_ftrs = self.model.fc.out_features
        #self.model.layer4=Identity()
        
        
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
 
        #specaugment
        self.transform = torchvision.transforms.RandomApply([ torchvision.transforms.Compose([T.TimeMasking(time_mask_param=80),
                                                                     T.FrequencyMasking(freq_mask_param=40),],)
                                                    ],
                                                    p=0.5)                            

    def batch_min_max(batch):
        batch = (batch-batch.min())/(batch.max()-batch.min())
        return batch

    def forward(self, x,augment=False):
        x = self.mel_scale(x)
        x = torchaudio.functional.amplitude_to_DB(x,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(x)) )
        x = x.squeeze(dim=1)
        x = ResLayer_attention.batch_min_max(x)

        if augment:
            x = self.transform(x)        
        x = torch.stack([x,x,x],axis=1)
        x = self.model(x)
        if self.tsne:
            return x
        x  = self.fc(x)
        return x


# def model_initialize(tsne=False):
#     model = ResLayer(tsne).cuda()
#     return model

# model=model_initialize(tsne=False)



# model = my_resnet18(drop_prob=0.3, block_size=18)
# #if you need pretrained weights
# model.load_state_dict(models.resnet18(pretrained=True).state_dict())


class Resnet34_wav(nn.Module):
    def __init__(self, mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        super(Resnet34_wav, self).__init__()
        # if "center=True" of stft, padding = win_len / 2

        #self.num_ftrs = 63

        self.res = ResLayer34()


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

class Resnet_wav_logspectro(nn.Module):
    def __init__(self, mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        super(Resnet_wav_logspectro, self).__init__()
        # if "center=True" of stft, padding = win_len / 2

        #self.num_ftrs = 63

        self.res = ResLayer()


        # self.spec = T.Spectrogram(n_fft=win_len,hop_length=hop_len,power=2)

        # self.mel_scale = T.MelScale(
        #     n_mels=mel_bins, sample_rate=16000, n_stft=win_len // 2 + 1
        #     )

        self.power_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        self.get_spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_len,
            hop_length=hop_len,
            center=True,
            pad_mode="constant",
            power=2.0)

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
        mel = self.get_spectrogram(x)[:,:100,:]

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


class ResLayer_wav_fusion_lstm(nn.Module):
    def __init__(self,mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        super(ResLayer_wav_fusion_lstm, self).__init__()
        self.wav_model = models.resnet18(pretrained=True).cuda() 
        self.egg_model = models.resnet18(pretrained=True).cuda() 
        self.num_ftrs = self.wav_model.fc.out_features
        hidden_size = 256
        #self.tsne = tsne
        self.lstm = nn.LSTM(input_size = 1,
                            hidden_size=hidden_size,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = True)
                            
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
                
        #self.fc = nn.Linear(2, 2),
        self.fc = nn.Sequential(       
            nn.Linear(hidden_size*2, 64),
                             nn.BatchNorm1d(64),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(64,50),
                             nn.BatchNorm1d(50),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(50,2)
                            )

    #@classmethod   
    def batch_min_max(batch):
        batch = (batch-batch.min())/(batch.max()-batch.min())
        return batch

    def forward(self, x_list, augment=False):
        wav = self.mel_scale(x_list[:,0,...])
        wav = wav.squeeze(1)
        wav = torchaudio.functional.amplitude_to_DB(wav,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(wav)) ) 
        wav = ResLayer_wav_fusion_lstm.batch_min_max(wav)        

        egg = self.mel_scale(x_list[:,1,...])
        egg = egg.squeeze(1)
        egg = torchaudio.functional.amplitude_to_DB(egg,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(egg)) )
        egg = ResLayer_wav_fusion_lstm.batch_min_max(egg)
                

        wav = torch.stack([wav,wav,wav],axis=1)
        egg = torch.stack([egg,egg,egg],axis=1)

        wav = self.wav_model(wav)
        egg = self.egg_model(egg)
        x = torch.concat([wav,egg]  ,axis=1)
        x = x.unsqueeze(2)
        x,(hidden_state,cell_state) = self.lstm(x)
        hidden_state = torch.cat([hidden_state[-1], hidden_state[-2]], dim=-1)
        # if self.tsne:
        #     return hidden_state

        x  = self.fc(hidden_state)
        return x



class MMTM(nn.Module):
  def __init__(self, dim_visual, dim_skeleton, ratio):
    super(MMTM, self).__init__()
    dim = dim_visual + dim_skeleton
    dim_out = int(2*dim/ratio)
    self.fc_squeeze = nn.Linear(dim, dim_out)

    self.fc_visual = nn.Linear(dim_out, dim_visual)
    self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

    # initialize
    # with torch.no_grad():
    #   self.fc_squeeze.apply(init_weights)
    #   self.fc_visual.apply(init_weights)
    #   self.fc_skeleton.apply(init_weights)

  def forward(self, visual, skeleton):
    squeeze_array = []
    for tensor in [visual, skeleton]:
      tview = tensor.view(tensor.shape[:2] + (-1,))
      squeeze_array.append(torch.mean(tview, dim=-1))
    squeeze = torch.cat(squeeze_array, 1)

    excitation = self.fc_squeeze(squeeze)
    excitation = self.relu(excitation)

    vis_out = self.fc_visual(excitation)
    sk_out = self.fc_skeleton(excitation)

    vis_out = self.sigmoid(vis_out)
    sk_out = self.sigmoid(sk_out)

    dim_diff = len(visual.shape) - len(vis_out.shape)
    vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

    dim_diff = len(skeleton.shape) - len(sk_out.shape)
    sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

    return visual * vis_out, skeleton * sk_out


class ResLayer_wav_fusion_mmtm(nn.Module):
    def __init__(self,mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        super(ResLayer_wav_fusion_mmtm, self).__init__()
        self.wav_model = models.resnet18(pretrained=True).cuda() 
        self.egg_model = models.resnet18(pretrained=True).cuda()

        self.mmtm1 = MMTM(64,64,4)
        self.mmtm2 = MMTM(128,128,4)      
        self.mmtm3 = MMTM(256,256,4)     
        self.mmtm4 = MMTM(512,512,4)  

        # self.wav_model = MyResNet18()
        # # if you need pretrained weights
        # self.wav_model.load_state_dict(models.resnet18(pretrained=True).state_dict())
        # self.egg_model = MyResNet18()
        # # if you need pretrained weights
        # self.egg_model.load_state_dict(models.resnet18(pretrained=True).state_dict())

        self.num_ftrs = self.wav_model.fc.out_features
                            
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
                
        #self.fc = nn.Linear(2, 2),
        self.fc = nn.Sequential(       
                            nn.Linear(self.num_ftrs*2, self.num_ftrs),
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

    #@classmethod   
    def batch_min_max(batch):
        batch = (batch-batch.min())/(batch.max()-batch.min())
        return batch

    def forward(self, x_list, augment=False):
        wav = self.mel_scale(x_list[:,0,...])
        wav = wav.squeeze(1)
        wav = torchaudio.functional.amplitude_to_DB(wav,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(wav)) ) 
        wav = ResLayer_wav_fusion_lstm.batch_min_max(wav)        

        egg = self.mel_scale(x_list[:,1,...])
        egg = egg.squeeze(1)
        egg = torchaudio.functional.amplitude_to_DB(egg,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(egg)) )
        egg = ResLayer_wav_fusion_lstm.batch_min_max(egg)
                

        wav = torch.stack([wav,wav,wav],axis=1)
        egg = torch.stack([egg,egg,egg],axis=1)


        ##### INPUT LAYER #####

        wav = self.wav_model.conv1(wav) # 64, 64, 151
        #print('conv1 : ',x.size())
        wav = self.wav_model.bn1(wav)
        wav = self.wav_model.relu(wav)
        wav = self.wav_model.maxpool(wav)
        #print('maxpool : ',x.size()) # 64, 32, 76

        egg = self.egg_model.conv1(egg) # 64, 64, 151
        #print('conv1 : ',x.size())
        egg = self.egg_model.bn1(egg)
        egg = self.egg_model.relu(egg)
        egg = self.egg_model.maxpool(egg)
        #print('maxpool : ',x.size()) # 64, 32, 76


        ##### FIRST RESIDUAL #####
        wav = self.wav_model.layer1(wav) # 64, 32, 76
        egg = self.egg_model.layer1(egg) # 64, 32, 76

        #### FIRST MMTM ####
        wav, egg = self.mmtm1(wav, egg)

        ##### SECOND RESIDUAL #####
        wav = self.wav_model.layer2(wav) # 128, 16, 38
        egg = self.egg_model.layer2(egg) # 128, 16, 38

        #### SECOND MMTM ####
        wav, egg = self.mmtm2(wav, egg)

        ##### THIRD RESIDUAL #####
        wav = self.wav_model.layer3(wav) # 256, 8, 19
        egg = self.egg_model.layer3(egg) # 256, 8, 19

        #### THIRD MMTM ####
        wav, egg = self.mmtm3(wav, egg)

        ##### FOURTH RESIDUAL #####
        wav = self.wav_model.layer4(wav) # 512, 4, 10
        egg = self.egg_model.layer4(egg) # 512, 4, 10

        #### FOURTH MMTM ####
        wav, egg = self.mmtm4(wav, egg)        

        wav = self.wav_model.avgpool(wav)
        egg = self.egg_model.avgpool(egg)
        #print(x.size())
        
        wav = torch.flatten(wav, 1)
        wav = self.wav_model.fc(wav)# 512

        egg = torch.flatten(egg, 1)
        egg = self.egg_model.fc(egg)# 512        

        x = torch.concat([wav,egg]  ,axis=1)

        x = torch.cat([wav,egg],axis=1)
        x = self.fc(x)
        return x


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

class Resnet_wav_latefusion_phrase_vowel(nn.Module):
    """
    phrase + vowel latefusion에 사용 concat이 된 데이터
    """
    def __init__(self, mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        super(Resnet_wav_latefusion_phrase_vowel, self).__init__()

        self.res_phrase = models.resnet18(pretrained=True).cuda()
        self.res_vowel = models.resnet18(pretrained=True).cuda()
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
                            nn.Linear(self.num_ftrs*2, self.num_ftrs),
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
        phrase = self.mel_scale(x_list[:,0,...])
        phrase = torchaudio.functional.amplitude_to_DB(phrase,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(phrase)) )
        phrase = (phrase-phrase.min())/(phrase.max()-phrase.min())
        phrase = torch.squeeze(phrase,dim=1)

        vowel = self.mel_scale(x_list[:,1,...])
        vowel = torchaudio.functional.amplitude_to_DB(vowel,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(vowel)) )
        vowel = (vowel-vowel.min())/(vowel.max()-vowel.min())
        vowel = torch.squeeze(vowel,dim=1)

        phrase = torch.stack([phrase,phrase,phrase],axis=1)
        vowel = torch.stack([vowel,vowel,vowel],axis=1)

        #print(out.size())
        phrase=self.res_phrase(phrase)
        vowel=self.res_vowel(vowel)

        out = torch.cat([phrase,vowel],axis=1)
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
        mel_phrase = (mel_phrase-mel_phrase.min())/(mel_phrase.max()-mel_phrase.min())
        mel_phrase = torch.squeeze(mel_phrase,dim=1)
        


        mel_a = self.mel_scale(x_list[:,1,...])
        mel_a = torchaudio.functional.amplitude_to_DB(mel_a,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel_a)) )
        mel_a = (mel_a-mel_a.min())/(mel_a.max()-mel_a.min())
        mel_a = torch.squeeze(mel_a,dim=1)

        mel_i = self.mel_scale(x_list[:,2,...])
        mel_i = torchaudio.functional.amplitude_to_DB(mel_i,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel_i)) )
        mel_i = (mel_i-mel_i.min())/(mel_i.max()-mel_i.min())
        mel_i = torch.squeeze(mel_i,dim=1)

        mel_u = self.mel_scale(x_list[:,3,...])
        mel_u = torchaudio.functional.amplitude_to_DB(mel_u,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel_u)) )
        mel_u = (mel_u-mel_u.min())/(mel_u.max()-mel_u.min())
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

class Resnet_wav_latefusion_all_testing(nn.Module):
    def __init__(self, mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        super(Resnet_wav_latefusion_all_testing, self).__init__()

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
        self.lstm = nn.LSTM(input_size=1, # 차원
                             hidden_size=1, #출력 수 
                             num_layers=1,
                             batch_first = True,
                             bidirectional=False,
                             bias=True)

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

        out = torch.cat([out_phrase,out_a,out_i,out_u],axis=1).unsqueeze(2)
        out,_ = self.lstm(out)
        out = out.squeeze()
        #print(out.size())
        #out = out[:,-1,:]
        out = self.fc(out)

        return out

class Resnet_wav_latefusion_all_attention(nn.Module):
    def __init__(self, mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        super(Resnet_wav_latefusion_all_attention, self).__init__()

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
        self.feature_to_score_phrase = nn.Sequential(
            nn.Linear(self.num_ftrs, self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, 3),
            )
        self.feature_to_score = nn.Sequential(
            nn.Linear(self.num_ftrs, self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, 1),
            )
    def attention_module(self,phrase,a,i,u):
        phrase_score = self.feature_to_score(phrase)
        a_score = self.feature_to_score(a)
        i_score = self.feature_to_score(i)
        u_score = self.feature_to_score(u)

        score = torch.softmax(torch.cat([phrase_score,a_score,i_score,u_score],dim=1),dim=1)
        return score #attention score

    def forward(self, x_list,train=True):
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
        
        #attention score. 
        score = self.attention_module(out_phrase,out_a,out_i,out_u)

        out_phrase = score[:,0].unsqueeze(dim=1)*out_phrase
        out_a = score[:,1].unsqueeze(dim=1)*out_a
        out_i = score[:,2].unsqueeze(dim=1)*out_i
        out_u = score[:,3].unsqueeze(dim=1)*out_u
        #print("phrase : ",score[:,0],"a : ",score[:,1],"i : ",score[:,2],"u : ",score[:,3])
        out = torch.cat([out_phrase, out_a, out_i, out_u],axis=1)
        out = self.fc(out)

        return out #,phrase_score

class Resnet_wav_latefusion_all_attention(nn.Module):
    def __init__(self, mel_bins=128,win_len=1024,n_fft=1024, hop_len=512):
        super(Resnet_wav_latefusion_all_attention, self).__init__()

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

        # self.fc = nn.Sequential(       
        #                     nn.Linear(self.num_ftrs*4, self.num_ftrs),
        #                     nn.BatchNorm1d(self.num_ftrs),
        #                     nn.ReLU(),
        #                     nn.Dropout(p=0.5),
        #                     nn.Linear(self.num_ftrs,128),
        #                     nn.BatchNorm1d(128),
        #                     nn.ReLU(),
        #                     nn.Dropout(p=0.5),
        #                     nn.Linear(128,64),
        #                     nn.BatchNorm1d(64),
        #                     nn.ReLU(),
        #                     nn.Dropout(p=0.5),
        #                     nn.Linear(64,2),
        #                     )
        self.fc = nn.Sequential(       
                            nn.Linear(self.num_ftrs+6,128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(128,64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,2),
                            )        
        self.fc_a = nn.Sequential(
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
        self.fc_i = nn.Sequential(
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
        self.fc_u = nn.Sequential(
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

        self.feature_to_score_phrase = nn.Sequential(
            nn.Linear(self.num_ftrs, self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, 3),
            )
        self.feature_to_score = nn.Sequential(
            nn.Linear(self.num_ftrs, self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, 1),
            )
    def attention_module(self,phrase,a,i,u):
        phrase_score = self.feature_to_score(phrase)
        a_score = self.feature_to_score(a)
        i_score = self.feature_to_score(i)
        u_score = self.feature_to_score(u)

        score = torch.softmax(torch.cat([phrase_score,a_score,i_score,u_score],dim=1),dim=1)
        return score #attention score
    
    def attention_module(self,vowel):
        score=torch.softmax(vowel,dim=1)
        return score*vowel

    def forward(self, x_list,train=True):
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
        out_a=self.fc_a(self.res_a(out_a))
        out_i=self.fc_i(self.res_i(out_i))
        out_u=self.fc_u(self.res_u(out_u))
        
        #attention score. 
        #score = self.attention_module(out_phrase,out_a,out_i,out_u)

        #out_phrase = out_phrase
        #out_a = self.attention_module(out_a)
        #out_i = self.attention_module(out_i)
        #out_u = self.attention_module(out_u)
        #print("phrase : ",score[:,0],"a : ",score[:,1],"i : ",score[:,2],"u : ",score[:,3])
        out = torch.cat([out_phrase, out_a, out_i, out_u],axis=1)
        out = self.fc(out)

        return out #,phrase_score


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













def model_initialize(model_name,spectro_run_config, mel_run_config, mfcc_run_config,tsne=False):
    if model_name=='msf':
        model = MSF(mfcc_run_config['n_mfcc']).cuda()
    elif model_name=='wav_res':
        model = Resnet_wav(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
        #model = Resnet_wav_temporal(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()      
    elif model_name=='wav_res_smile':
        model = Resnet_wav_smile(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
        #model = Resnet_wav_temporal(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()   
    elif model_name=='wav_res_phrase_eggfusion_lstm':
        model = ResLayer_wav_fusion_lstm(mel_bins=mel_run_config['n_mels'],
                                         win_len=mel_run_config['win_length'],
                                         n_fft=mel_run_config["n_fft"],
                                         hop_len=mel_run_config['hop_length']).cuda()
    elif model_name=='wav_res_phrase_eggfusion_mmtm':
        model = ResLayer_wav_fusion_mmtm(mel_bins=mel_run_config['n_mels'],
                                         win_len=mel_run_config['win_length'],
                                         n_fft=mel_run_config["n_fft"],
                                         hop_len=mel_run_config['hop_length']).cuda()
    elif model_name=='wav_res_time_attention':
        model = ResLayer_attention().cuda()
    elif model_name=='wav_res_splicing':
        model = Resnet_wav(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
        #model = Resnet34_wav(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
    elif model_name=='wav_res_mixup':
        model = Resnet_wav(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
        #model = Resnet34_wav(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()      
    elif model_name=='wav_res_latefusion':
        model = Resnet_wav_latefusion(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
    elif model_name=='wav_res_concat':
        model = Resnet_wav(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
    elif model_name=='wav_res_concat_latefusion': # vowel들 concat + latefusion
        model = Resnet_wav_latefusion(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
    elif model_name =='wav_res_concat_phrase_vowel':
        model = Resnet_wav_latefusion_phrase_vowel(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
    elif model_name =='wav_res_latefusion_phrase_vowel':
        model = Resnet_wav_latefusion_all(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
    elif model_name == 'wav_res_concat_allfusion':
        model = Resnet_wav_latefusion_all_testing(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
    elif model_name == 'wav_res_concat_allfusion_attention':
        model = Resnet_wav_latefusion_all_attention(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
    elif model_name == 'baseline':
        model = ResLayer().cuda()
    elif model_name == 'decomp':
        model = ResLayer().cuda()
    
        #############
        #아래부터는 ablation study
        ############
    if model_name == 'se_resnet18':
        model = se_resnet18(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()
    elif model_name == 'efficient_b0':
        model = efficinet_bo(mel_bins=mel_run_config['n_mels'],win_len=mel_run_config['win_length'],n_fft=mel_run_config["n_fft"],hop_len=mel_run_config['hop_length']).cuda()

    return model