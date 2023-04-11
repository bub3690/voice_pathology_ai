
import librosa
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
import scipy
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
import joblib


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
            {'C': param_range,},]
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





if __name__=='__main__':
    #vgg=ResLayer2()
    #print(vgg(torch.randn((2,3,128,301))).size())
    print('hello world')