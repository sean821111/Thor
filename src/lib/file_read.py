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

    return ppg_signal



def read_csv(full_path):
    ppg_list = []
    with open(full_path, newline='') as csvfile:
        rows  = csv.reader(csvfile)
        for row in rows:
            if row != []:
                ppg_list.append(float(row[0]))
              
    return ppg_list


# for stark format
def csv_reader(file_path, source='all'):
    for f in os.listdir(file_path):
        if re.findall(source, f):
            file_name = str(f)
    if 'file_name' in locals():
        ext = os.path.splitext(file_name)[-1].lower()

        if ext == '.csv':
            # Read csv
            full_path = os.path.join(file_path, file_name)
            f = open(full_path, "r", encoding='utf-8') 
            lines = f.readlines()
            dataDict = []
            
            # PPG format
            if source == 'ppg':
                head = ['G', 'R', 'IR', 'index']
                
            # ECG format
            elif source =='ecg':
                head = ['ecg', 'index']
                
            # All in one package
            else: 
                head = ['ecg', 'G', 'R', 'IR', 
                        'accX', 'accY', 'accZ', 
                        'gyroX', 'gyroY', 'gyroZ']
            
            dataDict = {r:[] for r in head}
            for line in lines:
                line = line.replace("\n", "")
                if line != "" :
                    # avoid \ufeff occur
                    _tmp = line.encode('utf-8').decode('utf-8-sig').split(",")
                    for col in range(len(head)):
                        dataDict[head[col]].append(float(_tmp[col]))
                        
            return dataDict
        else:
            raise ValueError("file extention '{}' not csv!".format(ext))
       
            
    else:
        raise ValueError("file source '{}' not found!".format(source))

# integrate for both txt and csv format
def load_ppg(file_path, source=None):
    if source == None:
        ppg = read_csv(file_path)
        return ppg
    else:
        for f in os.listdir(file_path):
            if re.findall(source, f):
                file_name = str(f)
        if 'file_name' in locals():  
            ext = os.path.splitext(file_name)[-1].lower()
            full_path = os.path.join(file_path, file_name)
            if ext == '.csv':
                ppg = read_csv(full_path)
            else:
                ppg = read_txt(full_path)
            return ppg
        else:
            raise ValueError("file source '{}' not found!".format(source))
        
        
# Thor+ reader
def _thor2_csv_format(full_file_path):
    f = open(full_file_path, "r", encoding='utf-8') 
    lines = f.readlines()
    dataDict = []


    firstrow = lines[0].replace("\n", "")
    # PPG format
    head = [v for v in firstrow.split(',')]
        

    dataDict = {r:[] for r in head}
    for line in lines[1:]:
        line = line.replace("\n", "")

        if line != "" :
            # avoid \ufeff occur
            _tmp = line.encode('utf-8').decode('utf-8-sig').split(",")
            for col in range(len(head)):
                dataDict[head[col]].append(float(_tmp[col]))
    return dataDict
def thor2_reader(file_path, source=None):
    # file path including file name
    if source == None:
        data = _thor2_csv_format(file_path)
        return data
    else:
        for f in os.listdir(file_path):
            if re.findall(source, f):
                file_name = str(f)
        if 'file_name' in locals():  
            ext = os.path.splitext(file_name)[-1].lower()
            full_path = os.path.join(file_path, file_name)
            data = _thor2_csv_format(full_path)
            return data
        else:
            raise ValueError("file source '{}' not found!".format(source))

# data protocol have 3 channels ACC and 1 chanel PPG 
def readcsv1(file_path):
    f = open(file_path, "r", encoding='utf-8') 

    lines = f.readlines()

    dataDict = []
    head = ['accX', 'accY', 'accZ', 'acc', 'G']
    dataDict = {r:[] for r in head}

    for line in lines:
        line = line.replace("\n", "")
        if line != "" :
            # avoid \ufeff occur
            _tmp = line.encode('utf-8').decode('utf-8-sig').split(",")
            
            for col in range(len(head)):
    
                # divide 2048 for normalize
                if col < 3:
                    dataDict[head[col]].append(float(_tmp[col])/2048.0)
                else:
                    dataDict[head[col]].append(float(_tmp[col]))

    return dataDict

# data protocol have 3 channels ACC and 2 chanel PPG (G and IR)
def readcsv2(file_path):
    dataDict = []
    head = ['accX', 'accY', 'accZ', 'acc', 'G', 'IR']
    dataDict = {r:[] for r in head}
    
    with open(file_path, newline='', encoding="utf-8") as csvfile:
        rows  = csv.reader(csvfile)
        for row in rows:
            if row != []:
                for col in range(len(head)):
                    # divide 2048 for normalize
                    if col < 3:
                        dataDict[head[col]].append(float(row[col])/2048.0)
                    else:
                        dataDict[head[col]].append(float(row[col]))

    return dataDict


# data protocol have 3 channels ACC and 4 chanel PPG (G1,G2,R and IR)
def readcsv3(file_path):
    dataDict = []
    head = ['accX', 'accY', 'accZ', 'acc', 'G1', 'G2', 'R', 'IR']
    dataDict = {r:[] for r in head}
    with open(file_path, newline='', encoding="utf-8-sig") as csvfile:
        rows  = csv.reader(csvfile)
        for row in rows:
            if row != []:
                for col in range(len(head)):
                    # divide 2048 for normalize accX, Y and Z
                    if col < 3:
                        dataDict[head[col]].append(float(row[col])/2048.0)
                    else:
                        dataDict[head[col]].append(float(row[col]))

    return dataDict

# 上岳
def readcsv_ECG(file_path):
    dataDict = []
    head = ['lead1', 'lead2', 'lead3']
    dataDict = {r:[] for r in head}
    
    with open(file_path, newline='', encoding="utf-8") as csvfile:
        rows  = csv.reader(csvfile)
        next(rows) #skip header
        for row in rows:
            if row != []:
                for col in range(len(head)):
                    dataDict[head[col]].append(float(row[col]))

    return dataDict
