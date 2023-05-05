
import librosa
import librosa
from sklearn.svm import SVC
from xgboost import XGBClassifier
import timm
from .Ablations import Custom_svm,Custom_tabnet





def model_initialize(model_name,mel_run_config,save_result=False,tsne=False):
    if model_name=='linear_svm':
        classifier = Custom_svm(kernel='linear',save_result=save_result)
    elif model_name=='radial_svm':
        classifier = Custom_svm(kernel='rbf',save_result=save_result)
    elif model_name=='polynomial_svm':
        classifier = Custom_svm(kernel='poly',save_result=save_result)
    elif model_name=='tabnet':
        classifier = Custom_tabnet()
    else:
        classifier = Custom_svm(kernel='linear',save_result=save_result)

    return classifier