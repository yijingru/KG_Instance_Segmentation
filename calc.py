import numpy as np
import os


def read_files(folder):
    output = []
    f = open(folder, "rb")
    for line in f:
        output.append(float(line.strip()))
    f.close()
    return output

filename = 'weights_kaggle/seg_ap_list.txt'

data = read_files(filename)
print(np.mean(data))
