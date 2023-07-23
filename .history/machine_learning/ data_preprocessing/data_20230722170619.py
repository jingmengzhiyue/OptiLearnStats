import numpy as np
import pandas as pd

def txt2pd(path):
    with open(path, 'r') as file:
        data = file.readlines()
        print(data)
        
        
if __name__=="__main__":
    txt2pd("machine_learning/dataset/datingTestSet.txt")
    