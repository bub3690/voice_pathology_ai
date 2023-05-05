
import librosa
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
import scipy
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
import joblib
import torch

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score

# 각 모델에서, 학습, validation까지 마치고 전달.






class Custom_svm():
    def __init__(self,kernel,save_result):
        self.kernel = kernel
        self.model = SVC(kernel=self.kernel, C=1, random_state=0,probability=save_result)

    def save_checkpoint(self,checkpoint_path):
        joblib.dump(self.model,checkpoint_path)

    def load_checkpoint(self,checkpoint_path):
        self.model = joblib.load(checkpoint_path)

    def train(self,train_x,train_y,valid_x,valid_y):
        
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [
            {'C': param_range},]
        X = np.concatenate((train_x,valid_x),axis=0) 
        y = train_y + valid_y 
        test_fold = [-1] * len(train_x) + [0] * len(valid_y)
        ps = PredefinedSplit(test_fold)

        grid_search = GridSearchCV(self.model, param_grid=param_grid, cv=ps,n_jobs=-1,verbose=1)
        grid_search.fit(X, y)

        self.model = grid_search.best_estimator_ 
        
        
        train_score = self.model.score(train_x,train_y)
        val_score = self.model.score(valid_x,valid_y)
        print('Finish train. Best params : ',grid_search.best_params_)
        
        return train_score,val_score
    
    def inference(self,test_x,test_y,save_result=False):
        test_score = self.model.score(test_x,test_y)
        y_pred = self.model.predict(test_x) # 예측치
        if save_result:
            y_probs = self.model.predict_proba(test_x)
            return test_score,test_y,y_pred,y_probs
        return test_score, test_y, y_pred

class Custom_tabnet():
    def __init__(self):
        #parameter for tabnet
        tabnet_params = {
                        "optimizer_fn":torch.optim.Adam,
                        "optimizer_params":dict(lr=2e-2),
                        "scheduler_params":{"step_size":50, # how to use learning rate scheduler
                                        "gamma":0.9},
                        "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                        "mask_type":'sparsemax', # "sparsemax"
                        "gamma" : 1.3 # coefficient for feature reusage in the masks
                        }
        #table
        
        self.model = TabNetClassifier(**tabnet_params)
    
    def save_checkpoint(self,checkpoint_path):
        joblib.dump(self.model,checkpoint_path)

    def load_checkpoint(self,checkpoint_path):
        self.model = joblib.load(checkpoint_path)
        
        
    def train(self,train_x,train_y,valid_x,valid_y):
        train_x,train_y,valid_x,valid_y = np.array(train_x),np.array(train_y),np.array(valid_x),np.array(valid_y)
        self.model.fit(train_x,train_y,eval_set=[(train_x,train_y),(valid_x,valid_y)],eval_name=['train', 'val'],max_epochs=150,patience=10,batch_size=32,eval_metric=['auc'])
        train_score = self.model.history['train_auc'][-1]
        val_score = self.model.history['val_auc'][-1]
        return train_score,val_score
    
    def inference(self,test_x,test_y,save_result=False):
        y_probs = self.model.predict_proba(test_x)
        y_pred = (y_probs[:,1]>0.5)*1
        test_y = [int(label=='pathology') for label in test_y]# pathology가 1, normal이 0
        
        y_pred = y_pred
        test_score = accuracy_score(test_y,y_pred)
        print(test_score)
        #y_pred = self.model.predict(test_x) # 예측치
        if save_result:
            #y_probs = self.model.predict_proba(test_x)
            return test_score,test_y,y_pred,y_probs
        return test_score, test_y, y_pred
        



if __name__=='__main__':
    #vgg=ResLayer2()
    #print(vgg(torch.randn((2,3,128,301))).size())
    print('hello world')