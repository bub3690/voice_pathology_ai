import pickle

class PhraseData():
    
    phrase_dict = dict()

    def __init__(self,phrase_path):
        #load
        
        with open(phrase_path,"rb") as fr:
            PhraseData.phrase_dict = pickle.load(fr)
        return