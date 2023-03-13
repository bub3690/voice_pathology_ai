import json

with open('./mel-spectrogram.json') as f:
    json_object = json.load(f)
print(json_object['1004'])