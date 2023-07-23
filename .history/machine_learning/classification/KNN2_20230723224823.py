import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

def txt2pd(path):
    scaler = MinMaxScaler()
    
    data = pd.read_csv(path, sep='\t', header=None, names=['travel', 'game', 'ice-cream', 'label'])
    
    data['label'] = data['label'].replace({'largeDoses': 1, 'smallDoses': 2, 'didntLike': 3})
    
    data_scaled = data.copy()
    data_scaled.iloc[:, :3] = scaler.fit_transform(data_scaled.iloc[:, :3])
    
    return data_scaled

path = "/root/code/OptiLearnStats/machine_learning/dataset/datingTestSet.txt"
data = txt2pd(path)
train, test = data[:int(0.9*len(data))], data[int(0.9*len(data)):]
train_data = train.iloc[:, :3]
test_data = test.iloc[:, :3]
test_label = test.iloc[:, 3]

dist = np.zeros((len(test), len(train)))
for i in range(len(test)):
    for j in range(len(train)):
        dist[i, j] = np.sum(np.abs(test_data.values[i] - train_data.values[j]))

k = 1
top_k_indices = np.argsort(dist, axis=1)[:, -k:]

predit = np.zeros((len(test), k))
for i in range(len(test)):
    predit[i, :] = train.loc[top_k_indices[i, :], 'label']

most_common = np.apply_along_axis(lambda row: Counter(row).most_common(1)[0][0], axis=1, arr=predit)

t = 0
for i in range(len(most_common)):
    if abs(test_label.values[i] - most_common[i]) < 0.1:
        t = t + 1

print(t / len(most_common))
