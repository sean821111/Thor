# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:44:17 2021

@author: 11008804
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import csv


def diff(lines):
    diffSignal=[]
    max=len(lines)-2
    const=1/(8*(1/512))
    for index in range (2,max) :
             diffSignal.append(
                const*(
                    -1*lines[index-2]
                    -2*lines[index-1]
                    +2*lines[index+1]
                    +1*lines[index+2]
                    )
                )

    return diffSignal

# diffSignal=diff(list)

def square (diffSignal):
    squaredSignal=[]

    for element in diffSignal :
        squaredSignal.append(element*element)


    return squaredSignal

# squaredSignal=square(diffSignal)

def smooth (squaredSignal):
    smoothedSignal=[]
    add=0
    for index in range (30,len(squaredSignal)-31) :
        for count in range (0,31) :
            add+=(1/31 *(squaredSignal[index-count]))
        smoothedSignal.append(add)
        add=0

    return smoothedSignal
def finadpick (autocorr):
    add = 0
    for element in autocorr:
        if(element > np.mean(autocorr)):
            add += 1
    return add

# smoothedSignal=smooth(squaredSignal)


def r_peak(ecg_sig):
    # ECG preprocessing
    diffSignal=diff(ecg_sig)
    
    squaredSignal=square(diffSignal)

    smoothedSignal=smooth(squaredSignal)
    
    data = smoothedSignal
    df = pd.DataFrame(data)
    df.columns = ['ECG']
    dataset = df

    hrw = 0.85
    fs = 500 

    mov_avg = dataset['ECG'].rolling(int(fs*hrw)).mean() #Calculate moving average

    avg_hr = (np.mean(dataset.ECG))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x*3.5 for x in mov_avg] #MAX30001
    
    dataset['ECG_rollingmean'] = mov_avg #Append the moving average to the dataframe

    # Mark regions of interest
    window = []
    peaklist = []
    listpos = 0 #We use a counter to move over the different data columns

    for datapoint in dataset.ECG:
        rollingmean = dataset.ECG_rollingmean[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1
        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
        else: #If signal drops below local mean -> determine highest point
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1
    return peaklist

def ecg_hr_estimate(ecg_sig):
    # ECG preprocessing
    diffSignal=diff(ecg_sig)
    
    squaredSignal=square(diffSignal)

    smoothedSignal=smooth(squaredSignal)
    
    data = smoothedSignal
    df = pd.DataFrame(data)
    df.columns = ['ECG']
    dataset = df

    hrw = 0.85
    fs = 500 

    mov_avg = dataset['ECG'].rolling(int(fs*hrw)).mean() #Calculate moving average

    avg_hr = (np.mean(dataset.ECG))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x*3.5 for x in mov_avg] #MAX30001
    
    dataset['ECG_rollingmean'] = mov_avg #Append the moving average to the dataframe

    # Mark regions of interest
    window = []
    peaklist = []
    listpos = 0 #We use a counter to move over the different data columns

    for datapoint in dataset.ECG:
        rollingmean = dataset.ECG_rollingmean[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1
        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
        else: #If signal drops below local mean -> determine highest point
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1

    ybeat = [dataset.ECG[x] for x in peaklist] #Get the y-value of all peaks for plotting purposes

    RR_list = []
    HR_est_list = []
    
    
    for i in range(len(peaklist)-1):
        RR_interval = (peaklist[i+1] - peaklist[i]) 
        HR_est = (fs/RR_interval)*60
        HR_est_list.append(HR_est)
    # HR_average = np.mean(HR_est_list)
    return HR_est_list


'''
For ECG R peak detector

'''
def adap_local_max(sample_rate,smoothedSignal):
      #find-peak
    medium_max_count = 0
    medium_min_count = 0
    peak_count =0
    valley_count = 0
    peak_pos =[]
    adaptive_windows_size_min = math.ceil(sample_rate*0.5/2)  #最小時窗一半
    adaptive_windows_size_max = math.ceil(sample_rate/2) #最大時窗一半
    adaptive_windows_size = adaptive_windows_size_min #初始設定windows大小
    for i in  range(len(smoothedSignal)):
        if ((i > adaptive_windows_size) and (i <len(smoothedSignal)-adaptive_windows_size)):
            if (adaptive_windows_size > adaptive_windows_size_max):
                adaptive_windows_size = adaptive_windows_size_max   
            #找峰值    
            for j in range(math.ceil(adaptive_windows_size +1)):
                if ((smoothedSignal[i] > smoothedSignal[i -j])  and (smoothedSignal[i] >= smoothedSignal[i + j]) ):
                    medium_max_count = medium_max_count + 1
                    #中間大於左右次數
                    if (medium_max_count == math.ceil(adaptive_windows_size)):
                        
                        #如 a = adaptive_windows_size 可知 ppg_data(x,1)為最大值
                        peak_pos.append(i)

                        if (len(peak_pos) > 1 ): #更新視窗寬度
                            adaptive_windows_size = math.ceil ((peak_pos [peak_count] - peak_pos [peak_count-1]) / 2) ;
                            
                        peak_count+=1 
                                                              
        medium_max_count = 0
        medium_min_count = 0
    
    #ppg_peak_dict = {'peak': ppg_peak, 'index': peak_pos}
    #ppg_valley_dict = {'valley': ppg_valley, 'index': ppg_valley_x}
    return  peak_pos