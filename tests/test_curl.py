import os
import ast
import requests

from time import time
from scipy.io import wavfile
from sklearn.metrics import classification_report, accuracy_score

SERVE_MODEL_URL = 'http://192.168.1.11:8080/api/infer/'
path = './data/gsc_v2.1/test/'

y_true = []
y_pred = []

for _class in ['active', 'non_active']:
    for item in os.listdir('{}/{}'.format(path, _class)):
        sr, data = wavfile.read('{}/{}/{}'.format(path, _class, item))

        data = {'value': data.tolist(), 'sr': sr}
        response = requests.post(SERVE_MODEL_URL, json=data)

        pred = ast.literal_eval(response.text)
        pred = pred.get('message', 0)  # return '0' or '1'
        y_pred.append(pred)

        if _class == 'active':
            y_true.append('1')
        else:
            y_true.append('0')

print(accuracy_score(y_true=y_true, y_pred=y_pred))        
print(classification_report(y_true, y_pred))

