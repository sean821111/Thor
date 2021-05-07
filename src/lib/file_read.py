import os, sys
import numpy as np 
import csv
import re
import scipy.signal as signal


'''
file reading library  
'''

def read_txt(file_path, p = 0):
    f = open(file_path, "r", encoding='utf-8') 

    lines = f.readlines()

    ppg_signal = []
    for line in lines:
        line = line.replace("\n", "")
        if line != "" :
            if p == 0:
                # for OSRAM 
                # avoid \ufeff occur
                _tmp = line.encode('utf-8').decode('utf-8-sig').split(",")[0]
                _tmp = float(_tmp)
            else:
                tmp = line.encode('utf-8').decode('utf-8-sig').split(",")[0]
                _tmp = float(_tmp)
                
            ppg_signal.append(_tmp)
            
    ppg_samples = np.arange(0,len(ppg_signal),1)

    return ppg_samples, ppg_signal



def read_csv(full_path):
    ppg_list = []
    with open(full_path, newline='') as csvfile:
        rows  = csv.reader(csvfile)
        for row in rows:
            if row != []:
                ppg_list.append(float(row[1]))
              
    ppg_sample_list = np.arange(0,len(ppg_list),1)
    return ppg_sample_list,ppg_list


# integrate for both txt and csv format
def load_ppg(file_path, source):
    for f in os.listdir(file_path):
        if re.findall(source, f):
            file_name = str(f)
    if 'file_name' in locals():  
        ext = os.path.splitext(file_name)[-1].lower()
        full_path = os.path.join(file_path, file_name)
        if ext == '.csv':
            x, y = read_csv(full_path)
        else:
            x, y = read_txt(full_path)
        return x,y
    else:
        raise ValueError("file source '{}' not found!".format(source))