import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def txt2pd(path):
    
    with open(path, 'r') as file:
        scaler = MinMaxScaler()

        data = pd.read_csv(path, sep='\t', header=None, names=['travel', 'game', 'ice-cream', 'label'])
        data['label'] = data['label'].replace({'largeDoses': 1, 'smallDoses': 2, 'didntLike': 3})

# 输出DataFrame
    print(data) 
        
    
        
        
if __name__=="__main__":
    txt2pd("machine_learning/dataset/datingTestSet.txt")
    