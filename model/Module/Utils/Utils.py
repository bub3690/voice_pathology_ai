import librosa
from Dataset.Data import PhraseData
import numpy as np
import torch
import torchaudio.transforms as T



def get_melspectro(path,config):
        sig = PhraseData.phrase_dict[ str(path)+'-phrase.wav'] 
        #sig = preemphasis(sig)

        length = config["sr"]*2 #sample rate *2 padding을 위한 파라미터 (하이퍼 파라미터로인해 사이즈는 계속 바뀐다.)
        pad1d = lambda a, i: a[0:i] if a.shape[0] > i else np.hstack((a, np.zeros((i-a.shape[0]))))        
        sig = pad1d(sig,length)        
        
        ###signal norm
        sig = (sig-sig.mean())/sig.std()
        ###
        
        mel_feature = librosa.feature.melspectrogram(y=sig,
                                                    sr=config["sr"],
                                                    # hyp param
                                                    n_mels = config["n_mels"],
                                                    n_fft = config["n_fft"],
                                                    win_length = config["win_length"],
                                                    hop_length = config["hop_length"],
                                                    fmax = config["f_max"]
                                                    )
        mel_feature = librosa.core.power_to_db(mel_feature,ref=np.max)
        return mel_feature

def get_logspectro(path,config,mel_config):
        sig = PhraseData.phrase_dict[ str(path)+'-phrase.wav'] 
        #sig = preemphasis(sig)

        length = config["sr"]*2 #sample rate *2 padding을 위한 파라미터 (하이퍼 파라미터로인해 사이즈는 계속 바뀐다.)
        pad1d = lambda a, i: a[0:i] if a.shape[0] > i else np.hstack((a, np.zeros((i-a.shape[0]))))        
        sig = pad1d(sig,length)        
        
        ###signal norm
        sig = (sig-sig.mean())/sig.std()
        ###
        stft = librosa.stft(sig, win_length=config["win_length"],
                                n_fft=config["n_fft"],
                                hop_length=config["hop_length"]
                                   )
        magnitude = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(magnitude)
        
        log_spectrogram = log_spectrogram[:mel_config["n_mels"],:]

        return log_spectrogram        
        

def get_mfcc(path,config):
        sig = PhraseData.phrase_dict[ str(path)+'-phrase.wav'] 
        #sig = preemphasis(sig)

        length = config["sr"]*2 #sample rate *2 padding을 위한 파라미터 (하이퍼 파라미터로인해 사이즈는 계속 바뀐다.)
        pad1d = lambda a, i: a[0:i] if a.shape[0] > i else np.hstack((a, np.zeros((i-a.shape[0]))))        
        sig = pad1d(sig,length)        
        
        ###signal norm
        sig = (sig-sig.mean())/sig.std()
        ###
        sig_torch=torch.tensor(sig, dtype=torch.float32)
        
        
        MFCC = T.MFCC(
                        sample_rate = config["sr"],
                        n_mfcc = config["n_mfcc"],
                        melkwargs={
                            'n_fft': config["n_fft"],
                            'n_mels': config["n_mels"],
                            'hop_length': config["hop_length"],
                            'mel_scale': config["mel_scale"],
                            'win_length' : config["win_length"],
                            'f_max': config["f_max"]
                        }
                    )
        
        MFCCs=MFCC(sig_torch)

        return MFCCs



def get_mean_std(X_path_list,Y_path_list,mode,spectro_run_config,mel_run_config,mfcc_run_config):
    spectro_mean = 0
    spectro_std = 0

    data_list = []

    if mode == 'melspectrogram':
        for x in X_path_list:
            data_list.append(get_melspectro(x,mel_run_config))
        data_list = np.array(data_list)
        spectro_mean = data_list.mean()
        spectro_std = data_list.std()
    elif mode == 'logspectrogram':
        for x in X_path_list:
            data_list.append(get_logspectro(x,spectro_run_config,mel_run_config))
        data_list = np.array(data_list)
        spectro_mean = data_list.mean()
        spectro_std = data_list.std()
    elif mode == 'mfcc':
        for x in X_path_list:
            data_list.append(get_mfcc(x,mfcc_run_config).numpy())
        data_list = np.array(data_list)
        spectro_mean = data_list.mean()
        spectro_std = data_list.std()

    return spectro_mean,spectro_std