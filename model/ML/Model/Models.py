
import librosa
import librosa
from sklearn.svm import SVC
from xgboost import XGBClassifier
import timm
from Ablations import linear_svm,radial_svm,polynomial_svm





def model_initialize(model_name,mel_run_config,tsne=False):
    if model_name=='linear_svm':
        classifier = linear_svm()
    elif model_name=='radial_svm':
        classifier = radial_svm()
    elif model_name=='polynomial_svm':
        classifier = polynomial_svm()
    else:
        classifier = linear_svm()

    return classifier