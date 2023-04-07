
import librosa
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
import scipy
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit



# 각 모델에서, 학습, validation까지 마치고 전달.


class linear_svm():
    def train(self,train_x,train_y,valid_x,valid_y):
        model_initialize = SVC(kernel='linear', C=1, random_state=0)
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [
            {'clf__C': param_range, 'clf__kernel': ['linear']},]
        X = train_x + valid_x 
        y = train_y + valid_y 
        test_fold = [-1] * len(train_x) + [0] * len(valid_y)
        ps = PredefinedSplit(test_fold)

        grid_search = GridSearchCV(model_initialize, param_grid=param_grid, cv=ps,n_jobs=-1)
        grid_search.fit(X, y)

        self.model = grid_search.best_estimator_ 
        
        
        train_score = self.model.score(train_x,train_y)
        val_score = self.model.score(valid_x,valid_y)
        
        return train_score,val_score
    
    def inference(self,test_x,test_y):
        test_score = self.model.score(test_x,test_y)
        y_pred = self.model.predict(test_x) # 예측치     
        y_probs = self.model.predict_proba(test_x)
        return test_score,test_y,y_pred,y_probs

class radial_svm():
    def train(self,train_x,train_y,valid_x,valid_y):
        model_initialize = SVC(kernel='rbf', C=1, random_state=0)
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [
            {'clf__C': param_range, 'clf__kernel': ['rbf']},]
        X = train_x + valid_x 
        y = train_y + valid_y 
        test_fold = [-1] * len(train_x) + [0] * len(valid_y)
        ps = PredefinedSplit(test_fold)

        grid_search = GridSearchCV(model_initialize, param_grid=param_grid, cv=ps,n_jobs=-1)
        grid_search.fit(X, y)

        self.model = grid_search.best_estimator_ 
        
        
        train_score = self.model.score(train_x,train_y)
        val_score = self.model.score(valid_x,valid_y)
        
        return train_score,val_score
    
    def inference(self,test_x,test_y):
        test_score = self.model.score(test_x,test_y)
        y_pred = self.model.predict(test_x) # 예측치     
        y_probs = self.model.predict_proba(test_x)
        return test_score,test_y,y_pred,y_probs



class polynomial_svm():
    def train(self,train_x,train_y,valid_x,valid_y):
        model_initialize = SVC(kernel='poly', C=1, random_state=0)
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [
            {'clf__C': param_range, 'clf__kernel': ['poly']},]
        X = train_x + valid_x 
        y = train_y + valid_y 
        test_fold = [-1] * len(train_x) + [0] * len(valid_y)
        ps = PredefinedSplit(test_fold)

        grid_search = GridSearchCV(model_initialize, param_grid=param_grid, cv=ps,n_jobs=-1)
        grid_search.fit(X, y)

        self.model = grid_search.best_estimator_ 
        
        
        train_score = self.model.score(train_x,train_y)
        val_score = self.model.score(valid_x,valid_y)
        
        return train_score,val_score
    
    def inference(self,test_x,test_y):
        test_score = self.model.score(test_x,test_y)
        y_pred = self.model.predict(test_x) # 예측치     
        y_probs = self.model.predict_proba(test_x)
        return test_score,test_y,y_pred,y_probs






if __name__=='__main__':
    #vgg=ResLayer2()
    #print(vgg(torch.randn((2,3,128,301))).size())
    print('hello world')