import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def txt2pd(path):
    
    with open(path, 'r') as file:
        scaler = MinMaxScaler()
        
        data = pd.read_csv(path, sep='\t', header=None, names=['travel', 'game', 'ice-cream', 'label'])
        
        data['label'] = data['label'].replace({'largeDoses': 1, 'smallDoses': 2, 'didntLike': 3})
        
        data_scaled = data.copy()
        data_scaled.iloc[:, :3] = scaler.fit_transform(data_scaled.iloc[:, :3])

# 输出缩放后的DataFrame
        
        # data_scaled = pd.DataFrame(scaler.fit_transform(data_slice), columns=data.columns)
        # data = pd.concat(data_scaled, data[:,-1])
# 输出DataFrame
    return data_scaled
        
    
        
        
if __name__=="__main__":
    txt2pd("machine_learning/dataset/datingTestSet.txt")
    