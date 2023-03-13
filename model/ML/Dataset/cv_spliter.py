
import os
import pandas as pd
import random #데이터 shuffle 사용
from glob import glob
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split # train , test 분리에 사용.
import numpy as np

def cv_spliter(random_state,file_path):
    """
    5 CV data spliter
    Speark independence
    return : 
    
    """

    # # 데이터 나누기 - Stratified KFold
    # 
    # - pathology : 1194 / healthy : 634 / 총 1828
    # - k = 5


    # ## 1. test/ train 나누기
    # 


    speaker_data=pd.read_excel(file_path)


    pathology = speaker_data[speaker_data['PATHOLOGY']=='p']['SPEAKER'].unique().tolist()
    healthy = speaker_data[speaker_data['PATHOLOGY']=='n']['SPEAKER'].unique().tolist()
    print(len(pathology))
    print(len(healthy))


    #겹치는 speaker는 곱하기 100을 해준다.
    #3명이 겹친다.
    changed_patients = list(set(healthy) & set(pathology))

    for patient in changed_patients:
        temp=pathology[pathology.index(patient)]*100
        pathology[pathology.index(patient)] = temp
        


    #train test 나누기
    

    X = pathology+healthy # path 데이터 합
    print("총 데이터수 : ",len(X))
    Y = [] # 라벨
    for idx,x in enumerate(X):
        if idx<len(pathology):
            Y.append("pathology")
        else:
            Y.append("healthy")

    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y, random_state=random_state) #456
    #stratify를 넣어서, test에도 라벨별 잘 분류되게 한다.

    print("---")
    print("훈련 셋 : ",len(Y),Counter(Y))
    print("테스트 셋 : ",len(Y_test),Counter(Y_test))
    print("---")




    # ## 2. stratified k-fold


    #1. train, test 나누기
    #stratified kfold
    skf_iris = StratifiedKFold(n_splits=5,shuffle=True,random_state=456)
    cnt_iter = 0

    X_train_list = [] #데이터 셋 보관
    Y_train_list = []

    X_valid_list = []
    Y_valid_list = []

    for train_idx, test_idx in skf_iris.split(X,Y):
        
        #split으로 반환된 인덱스를 이용하여, 학습 검증용 테스트 데이터 추출
        cnt_iter += 1
        X_train, X_valid = [X[idx] for idx in train_idx.tolist() ], [X[idx] for idx in test_idx.tolist() ]
        Y_train, Y_valid = [Y[idx] for idx in train_idx.tolist() ], [Y[idx] for idx in test_idx.tolist() ]
        
        X_train_list.append(X_train)
        X_valid_list.append(X_valid)
        
        Y_train_list.append(Y_train)
        Y_valid_list.append(Y_valid)
        
        
        #학습 및 예측
        
        label_train = Y_train
        label_test = Y_valid
        unique_train, train_counts = np.unique(label_train, return_counts = True)
        unique_test, test_counts = np.unique(label_test, return_counts = True)

        uniq_cnt_train = dict(zip(unique_train, train_counts))
        uniq_cnt_test = dict(zip(unique_test, test_counts))
        
        
        
        print('교차 검증 : {}'.format(cnt_iter))
        print('학습 레이블 데이터 분포 : \n', uniq_cnt_train)
        print('검증 레이블 데이터 분포 : \n', uniq_cnt_test,'\n')




    # speaker to voice

    label_changer = dict({"healthy":"n","pathology":"p"})


    all_train_record_list = []
    all_valid_record_list = []
    all_test_record_list = []

    all_train_label_list = []
    all_valid_label_list = []
    all_test_label_list = []

    print("train. speaker to voice")
    #train
    for fold_idx,fold in enumerate(X_train_list):
        fold_record=[]
        fold_y_record=[]
        for idx,speaker in enumerate(fold):
            record_list = speaker_data[ (speaker_data['SPEAKER']==speaker) & (speaker_data['PATHOLOGY']==label_changer[Y_train_list[fold_idx][idx]])]['RECORDING'].tolist()
            ####
            if record_list == []:
                # speaker가 healthy, pathology 모두 있는 경우
                #print(speaker)
                speaker = speaker // 100
                record_list = speaker_data[(speaker_data['SPEAKER']==speaker) & (speaker_data['PATHOLOGY']==label_changer[Y_train_list[fold_idx][idx]] ) ]['RECORDING'].tolist()
                #print(record_list)

            # 중복화자 한명 남기기 실험. 2023.01.18
            #record_list = [record_list[0],]
            

            label_list = [ Y_train_list[fold_idx][idx] ] * len(record_list)       
            fold_record += record_list
            fold_y_record += label_list
        all_train_record_list.append(fold_record)
        all_train_label_list.append(fold_y_record)

    print("valid. speaker to voice")
    #valid
    for fold_idx,fold in enumerate(X_valid_list):
        fold_record=[]
        fold_y_record=[]
        for idx,speaker in enumerate(fold):
            record_list = speaker_data[ (speaker_data['SPEAKER']==speaker) & (speaker_data['PATHOLOGY']==label_changer[Y_valid_list[fold_idx][idx]]) ]['RECORDING'].tolist()
            if record_list == []:
                # speaker가 healthy, pathology 모두 있는 경우
                #print(speaker)
                speaker = speaker // 100
                record_list = speaker_data[(speaker_data['SPEAKER']==speaker) & (speaker_data['PATHOLOGY']==label_changer[Y_valid_list[fold_idx][idx]] ) ]['RECORDING'].tolist()
                #print(record_list)
            label_list = [ Y_valid_list[fold_idx][idx] ] * len(record_list)
            
            fold_record += record_list
            fold_y_record += label_list
        all_valid_record_list.append(fold_record)
        all_valid_label_list.append(fold_y_record)

    print("test. speaker to voice")
    #test
    fold_record=[]
    fold_y_record=[]
    for idx,speaker in enumerate(X_test):
        record_list = speaker_data[(speaker_data['SPEAKER']==speaker) & (speaker_data['PATHOLOGY']==label_changer[Y_test[idx]] )]['RECORDING'].tolist()
        if record_list == []:
            # speaker가 healthy, pathology 모두 있는 경우
            #print(speaker)
            speaker = speaker // 100
            record_list = speaker_data[(speaker_data['SPEAKER']==speaker) & (speaker_data['PATHOLOGY']==label_changer[Y_test[idx]] ) ]['RECORDING'].tolist()
            #print(record_list)
        label_list = [ Y_test[idx] ] * len(record_list)
        fold_record += record_list
        fold_y_record += label_list
    all_test_record_list = fold_record
    all_test_label_list = fold_y_record


    X_train_list = all_train_record_list
    X_valid_list = all_valid_record_list
    X_test = all_test_record_list

    Y_train_list = all_train_label_list
    Y_valid_list = all_valid_label_list
    Y_test = all_test_label_list






    # ## 3. random over sampling


    #2. random over sampling
    for i in range(5):
        X_temp = np.array(X_train_list[i]).reshape(-1,1)#각 데이터를 다 행으로 넣음. (1194,1)
        #Y = np.array(Y)
        ros = RandomOverSampler(random_state = 123)
        X_res,Y_res = ros.fit_resample(X_temp,Y_train_list[i])
        
        print("\n fold{} ".format(i))
        print('before dataset shape {}'.format(Counter(Y_train_list[i])) )
        print('Resampled dataset shape {}'.format(Counter(Y_res)) )   
        
        #원래대로 돌리기
        X_res=X_res.reshape(1, -1)
        X_train_list[i]=X_res[0].tolist()
        Y_train_list[i]=Y_res

    return X_train_list,X_valid_list,X_test,Y_train_list,Y_valid_list,Y_test


if __name__=='main':
    cv_spliter()