import numpy as np
import sys
from collections import Counter
sys.path.append('../ data_preprocessing/')
from data import txt2pd

path = "/root/code/OptiLearnStats/machine_learning/dataset/datingTestSet.txt"
data = txt2pd(path)
train, test = data[:int(0.8*len(data))],data[int(0.8*len(data)):]
train_data = train.iloc[:, :3]
test_data = test.iloc[:, :3]
dist = np.zeros((len(test),len(train)))
for i in range(len(test)):
    for j in range(len(train)):
        dist[i, j] = np.sum( np.abs(test_data.values[i] - train_data.values[j]))
top_k_indices = np.argsort(dist, axis=1)[:, -7:]
# print(top_k_indices)
predit = np.zeros((len(test), 7))
for i in range(len(test)):
    predit[i,:] = column_4_data = train.loc[top_k_indices[i,:], 'label']
most_common = np.apply_along_axis(lambda row: Counter(row).most_common(1)[0][0], axis=1, arr=predit)
t=0
for i in range(len(most_common)):
    if test.loc[i, 'label'] == most_common[i]:
        t = t+1
print(t/len(most_common))

# 输出结果
print(train['label'])
print(most_common)
# print(predit)s