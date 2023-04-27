import librosa
from Dataset.Data import PhraseData,OpensmileData,GlottalData
import pandas as pd
import numpy as np
import torch
import torchaudio.transforms as T
import opensmile
from sklearn.preprocessing import PowerTransformer,QuantileTransformer,StandardScaler,MinMaxScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import os
import pickle
import scipy

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif





def statistical_feature(feature_vec):
    mean = np.mean(feature_vec)
    std = np.std(feature_vec) 
    maxv = np.amax(feature_vec) 
    minv = np.amin(feature_vec) 
    skew = scipy.stats.skew(feature_vec)
    kurt = scipy.stats.kurtosis(feature_vec)
    q1 = np.quantile(feature_vec, 0.25)
    median = np.median(feature_vec)
    q3 = np.quantile(feature_vec, 0.75)
    mode = scipy.stats.mode(feature_vec)[0][0]
    iqr = scipy.stats.iqr(feature_vec)
    
    return [mean, std, maxv, minv, median, skew, kurt, q1, q3, mode, iqr]


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

def get_smile(path,config,num_workers=0):
        smile_data = OpensmileData.opensmile_dict[ str(path)+'-phrase.wav'] 
        return smile_data

def get_glottal(data_list,config):
        glottal_data = GlottalData.glottal_table.loc[GlottalData.glottal_table['RECORDING'].isin( data_list )].iloc[:,1:]
        return glottal_data


def get_scaler(X_path_list,Y_path_list,mode,spectro_run_config,mel_run_config,mfcc_run_config,num_workers=0):
    data_list = []
    X_path_list = list(set(X_path_list))
    if mode == 'smile':
        #scaler = PowerTransformer(method='yeo-johnson',standardize=True)
        scaler = QuantileTransformer(output_distribution='uniform')
        for x in tqdm(X_path_list):
            data_list.append(get_smile(x,mfcc_run_config,num_workers=num_workers).to_numpy().squeeze())
        data_list = np.array(data_list)
        scaler.fit(data_list)
        OpensmileData.scaler_list.append(scaler)
    elif mode == 'glottal':
        # glottal feature는 적절한 scaler를  찾아야한다.

        scaler = QuantileTransformer(output_distribution='uniform')
        data_list.append(get_glottal(X_path_list,mfcc_run_config).to_numpy())
        data_list = np.array(data_list)
        scaler.fit(data_list)
        GlottalData.scaler_list.append(scaler)
    
    return scaler
        


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



def save_result(all_filename, all_prediction, all_answers,all_probs,speaker_file_path_abs,args):
    fold_excel = []
    for i in range(5):
        fold_excel.append(pd.DataFrame({'filename':all_filename[i],
                    'prediction':[data.cpu().numpy().item() for data in all_prediction[i]],
                    'answer':[ data.cpu().numpy().item() for data in all_answers[i]],
                    'prob':[ data.cpu().numpy().item() for data in all_probs[i]],
                    'fold':i+1}))
    #print(fold_excel)
    #import pdb;pdb.set_trace()
    
    fold_excel_all=pd.concat(fold_excel,axis=0)
    
    answer_paper=pd.read_excel(speaker_file_path_abs)
    answer_paper['RECORDING']=answer_paper['RECORDING'].values.astype(str)
    #answer_paper[['RECORDING','DETAIL','AGE']]
    merge_left = pd.merge(fold_excel_all,answer_paper[['RECORDING','DETAIL','AGE','DIAG']], how='left', left_on='filename', right_on='RECORDING')
    merge_left.drop(['RECORDING'],axis=1,inplace=True)
    merge_left['result']=merge_left['prediction']==merge_left['answer']
    merge_left['filename']=merge_left['filename'].values.astype(int)
    merge_left = merge_left[['filename','fold','AGE','DETAIL','prediction','answer','prob','DIAG','result']]

    data_subset = 'organics'
    if args.data_subset==0:
        data_subset = 'alldata'
    elif args.data_subset==1:
        data_subset = 'organics'

    excel_name = '../../voice_data/results/'+args.model+'_'+args.dataset+'_seed_'+str(args.seed)+'_dataprobs_'+str(args.data_probs)+'_'+data_subset+'_speaker.xlsx'
    excel_name = os.path.abspath(excel_name)
    print(os.getcwd())
    print(excel_name)
    #excel_name = './'+args.model+'_'+args.dataset+'_seed_'+str(args.seed)+'_organics_speaker.xlsx'
    merge_left.to_excel(excel_name,index=False)


class FeatureSelector():
    def __init__(self,):
        self.selectors = [] #fold에 따라 다른 selector
     
    def feature_selection(self,train_x,train_y,k=512):
        print("Feature selection")
        selector = SelectKBest(mutual_info_classif, k=k)
        train_x_new = selector.fit_transform(train_x, train_y)
        self.selectors.append(selector)
        return train_x_new

    def feature_selection_inference(self,valid_x,fold,k=512):
        print("Feature selection. inference")
        valid_x_new = self.selectors[fold].transform(valid_x)    
        return valid_x_new

class PostScaler():
    """
    deeplearning feature, smile feature를 다시 scaling 하는 클래스
    """

    def __init__(self,):
        self.scalers = [] #fold에 따라 다른 scaler
    
    def post_scaling(self,train_x):
        print("Post scaling")
        scaler = StandardScaler()
        train_x_new = scaler.fit_transform(train_x)
        self.scalers.append(scaler)
        return train_x_new
    
    def post_scaling_inference(self,valid_x,fold):
        print("Post scaling. inference")
        valid_x_new = self.scalers[fold].transform(valid_x)    
        return valid_x_new

if __name__ =='__main__':
    print(os.getcwd())
