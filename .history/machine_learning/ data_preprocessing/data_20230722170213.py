import numpy as np
import pandas as pd

def txt2pd(path):
    with open(path, 'r') as file:
        data = file.readlines()
        print(data)
        