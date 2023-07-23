import numpy as np
import pandas as pd

def txt2pd(path):
    with open(path, 'r') as file:
        data = pd.read_csv(path, sep='\t', header=None, names=['col1', 'col2', 'col3', 'col4'])

# 输出DataFrame
    print(data) 
        
    
        
        
if __name__=="__main__":
    txt2pd("machine_learning/dataset/datingTestSet.txt")
    