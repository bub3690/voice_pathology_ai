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


###
#기본형
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

class Layer_wav(nn.Module):
    def __init__(self, mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,LAYER=ResLayer()):
        #mel_bins=128,win_len=1024,n_fft=1024, hop_len=512
        super(Layer_wav, self).__init__()
        # if "center=True" of stft, padding = win_len / 2

        #self.num_ftrs = 63

        self.res = LAYER.cuda()


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

    def batch_min_max(batch):
        batch = (batch-batch.min())/(batch.max()-batch.min())
        return batch
    

    def forward(self, x,augment=False):
        mel = self.mel_scale(x)
        
        mel = torchaudio.functional.amplitude_to_DB(mel,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(mel)) )
        mel = torch.squeeze(mel,dim=1)        
        mel = Layer_wav.batch_min_max(mel)


        out = torch.stack([mel,mel,mel],axis=1)
        #print(out.size())
        out=self.res(out)
        return out


###


#seresnet
def se_resnet18(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    model = timm.create_model('legacy_seresnet18',num_classes=1000,pretrained=True)
    print(model)
    num_ftrs=model.last_linear.out_features
    
    model.fc = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,2)
                        )
    
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model

def se_resnet34(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    model = timm.create_model('legacy_seresnet34',num_classes=1000,pretrained=True)
    print(model)
    num_ftrs=model.last_linear.out_features
    
    model.fc = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,2)
                        )
    
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model

def se_resnet50(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    model = timm.create_model('legacy_seresnet50',num_classes=1000,pretrained=True)
    print(model)
    num_ftrs=model.last_linear.out_features
    
    model.fc = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,2)
                        )
    
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model

def se_resnet101(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    model = timm.create_model('legacy_seresnet101',num_classes=1000,pretrained=True)
    print(model)
    num_ftrs=model.last_linear.out_features
    
    model.fc = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,2)
                        )
    
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model



#####
def xception(num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    model = timm.create_model('xception',num_classes=1000,pretrained=True)
    #print(model)
    num_ftrs=model.fc.in_features
    model.fc = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,2)
                        )

    return model


###



###
def efficient_b0(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    num_ftrs = 1000
    model = models.efficientnet_b0(pretrained=True,num_classes=num_ftrs)
    #print(model)
    classifier = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,num_classes)
                        )
    model.classifier = nn.Sequential(*list(model.classifier) + [classifier])
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model

def efficient_b1(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    num_ftrs = 1000
    model = models.efficientnet_b1(pretrained=True,num_classes=num_ftrs)
    #print(model)
    classifier = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,num_classes)
                        )
    model.classifier = nn.Sequential(*list(model.classifier) + [classifier])
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model

def efficient_b2(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    num_ftrs = 1000
    model = models.efficientnet_b2(pretrained=True,num_classes=num_ftrs)
    #print(model)
    classifier = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,num_classes)
                        )
    model.classifier = nn.Sequential(*list(model.classifier) + [classifier])
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model


def efficient_b3(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    num_ftrs = 1000
    model = models.efficientnet_b3(pretrained=True,num_classes=num_ftrs)
    #print(model)
    classifier = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,num_classes)
                        )
    model.classifier = nn.Sequential(*list(model.classifier) + [classifier])
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model



###res

def resnet34(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(64,50),
                        nn.BatchNorm1d(50),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(50,2)
                        )
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model

def resnet50(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(64,50),
                        nn.BatchNorm1d(50),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(50,2)
                        )
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model

def resnet101(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    model = models.resnet101(pretrained=True)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(64,50),
                        nn.BatchNorm1d(50),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(50,2)
                        )
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model


## densenet
def densenet_121(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = timm.create_model('densenet121',num_classes=1000,pretrained=True)
    #print(model)
    num_ftrs=model.classifier.in_features
    
    model.fc = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,2)
                        )
    
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model



## alexnet
def alexnet(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    model = models.alexnet(pretrained=True)
    num_ftrs = 1000

    classifier = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,num_classes)
                        )
    model.classifier = nn.Sequential(*list(model.classifier) + [classifier])

    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model


####
#else


def mixnet_l(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    model = timm.create_model(model_name='mixnet_l',pretrained=True)
    #print(model)
    num_ftrs=model.classifier.in_features
    model.classifier = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,2)
                        )
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model


def mixerb16(mel_bins=128,win_len=1024,n_fft=1024, hop_len=512,num_classes=2):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # fc layer 추가해서 고쳐보기
    model = timm.create_model(model_name='mixer_b16_224',pretrained=True)
    #print(model)
    num_ftrs=model.head.in_features
    model.head = nn.Sequential(       
        nn.Linear(num_ftrs, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(64,50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(50,2)
                        )
    base_model = Layer_wav(mel_bins=mel_bins,win_len=win_len,n_fft=n_fft, hop_len=hop_len,LAYER=model)

    return base_model