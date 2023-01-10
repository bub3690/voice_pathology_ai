import pickle
import os

def make_data(dataset):
    if dataset == 'phrase':
        phras_file_path = "../../voice_data/organics_ver2/phrase_dict_ver2.pickle"
        phras_file_path_abs = os.path.abspath(phras_file_path)
    elif dataset == 'a_h':
        phras_file_path = "../../voice_data/organics_ver2/a_high_dict_ver2.pickle"
        phras_file_path_abs = os.path.abspath(phras_file_path)
    elif dataset == 'a_l':
        phras_file_path = "../../voice_data/organics_ver2/a_low_dict_ver2.pickle"
        phras_file_path_abs = os.path.abspath(phras_file_path)
    elif dataset == 'a_n':
        phras_file_path = "../../voice_data/organics_ver2/a_neutral_dict_ver2.pickle"
        phras_file_path_abs = os.path.abspath(phras_file_path)
    elif dataset == 'a_fusion':
        fusion_file_list = [os.path.abspath("../../voice_data/organics_ver2/a_high_dict_ver2.pickle"),
                            os.path.abspath("../../voice_data/organics_ver2/a_low_dict_ver2.pickle"),
                            os.path.abspath("../../voice_data/organics_ver2/a_neutral_dict_ver2.pickle")
                            ]
        fusion_name_list = ["a_h","a_l","a_n"]
    elif dataset == 'i_h':
        phras_file_path = "../../voice_data/organics_ver2/i_high_dict_ver2.pickle"
        phras_file_path_abs = os.path.abspath(phras_file_path)
    elif dataset == 'i_l':
        phras_file_path = "../../voice_data/organics_ver2/i_low_dict_ver2.pickle"
        phras_file_path_abs = os.path.abspath(phras_file_path)
    elif dataset == 'i_n':
        phras_file_path = "../../voice_data/organics_ver2/i_neutral_dict_ver2.pickle"
        phras_file_path_abs = os.path.abspath(phras_file_path)
    elif dataset == 'i_fusion':
        fusion_file_list = [os.path.abspath("../../voice_data/organics_ver2/i_high_dict_ver2.pickle"),
                            os.path.abspath("../../voice_data/organics_ver2/i_low_dict_ver2.pickle"),
                            os.path.abspath("../../voice_data/organics_ver2/i_neutral_dict_ver2.pickle")
                            ]
        fusion_name_list = ["i_h","i_l","i_n"]
    elif dataset == 'u_h':
        phras_file_path = "../../voice_data/organics_ver2/u_high_dict_ver2.pickle"
        phras_file_path_abs = os.path.abspath(phras_file_path)
    elif dataset == 'u_l':
        phras_file_path = "../../voice_data/organics_ver2/u_low_dict_ver2.pickle"
        phras_file_path_abs = os.path.abspath(phras_file_path)
    elif dataset == 'u_n':
        phras_file_path = "../../voice_data/organics_ver2/u_neutral_dict_ver2.pickle"
        phras_file_path_abs = os.path.abspath(phras_file_path)
    elif dataset == 'u_fusion':
        fusion_file_list = [os.path.abspath("../../voice_data/organics_ver2/u_high_dict_ver2.pickle"),
                            os.path.abspath("../../voice_data/organics_ver2/u_low_dict_ver2.pickle"),
                            os.path.abspath("../../voice_data/organics_ver2/u_neutral_dict_ver2.pickle")
                            ]
        fusion_name_list = ["u_h","u_l","u_n"]
    elif dataset == 'all_fusion':
        fusion_file_list = [
                            os.path.abspath("../../voice_data/organics_ver2/phrase_dict_ver2.pickle"),

                            os.path.abspath("../../voice_data/organics_ver2/a_high_dict_ver2.pickle"),
                            os.path.abspath("../../voice_data/organics_ver2/a_low_dict_ver2.pickle"),
                            os.path.abspath("../../voice_data/organics_ver2/a_neutral_dict_ver2.pickle"),

                            os.path.abspath("../../voice_data/organics_ver2/i_high_dict_ver2.pickle"),
                            os.path.abspath("../../voice_data/organics_ver2/i_low_dict_ver2.pickle"),
                            os.path.abspath("../../voice_data/organics_ver2/i_neutral_dict_ver2.pickle"),

                            os.path.abspath("../../voice_data/organics_ver2/u_high_dict_ver2.pickle"),
                            os.path.abspath("../../voice_data/organics_ver2/u_low_dict_ver2.pickle"),
                            os.path.abspath("../../voice_data/organics_ver2/u_neutral_dict_ver2.pickle"),

                            ]
        fusion_name_list = ["phrase","a_h","a_l","a_n","i_h","i_l","i_n","u_h","u_l","u_n"]
        
    else:
        phras_file_path = "../../voice_data/organics/phrase_sig_dict.pickle"
        phras_file_path_abs = os.path.abspath(phras_file_path)


    if dataset.split('_')[1] in 'fusion':
        #fusion data 구성
        print("데이터 로드 " + dataset)
        data_instance = FusionData(fusion_file_list) #class에 데이터를 담아준다.
    else:
        print("데이터 로드 "+dataset)
        data_instance = PhraseData(phras_file_path_abs) #class에 데이터를 담아준다.
    return data_instance




class PhraseData():
    
    phrase_dict = dict()

    dict_list = [] # fusion을 위한 dict 리스트

    def __init__(self,phrase_path):
        #load
        
        with open(phrase_path,"rb") as fr:
            PhraseData.phrase_dict = pickle.load(fr)
        return

class FusionData():

    dict_list = [] # fusion을 위한 dict 리스트

    def __init__(self,path_list):
        #load
        
        for wav_path in path_list:
            with open(wav_path,"rb") as fr:
                FusionData.dict_list.append(pickle.load(fr))
        return