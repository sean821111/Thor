
import numpy as np
import math
import scipy.signal as signal
from scipy.signal import butter, iirnotch, lfilter,filtfilt
import pandas as pd


## A high pass filter allows frequencies higher than a cut-off value
def butter_highpass(cutoff, sr, order):
    nyq = 0.5*sr
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a
## A low pass filter allows frequencies lower than a cut-off value
def butter_lowpass(cutoff, sr, order):
    nyq = 0.5*sr
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a
def notch_filter(cutoff, q, sr):
    nyq = 0.5*sr
    freq = cutoff/nyq
    b, a = iirnotch(freq, q)
    return b, a

def highpass(data, sr, order, cutoff_high=1):
    b,a = butter_highpass(cutoff_high, sr, order=order)
    x = filtfilt(b,a,data)
    return x

def lowpass(data, sr, order, cutoff_low=40):
    b,a = butter_lowpass(cutoff_low, sr, order=order)
    y = filtfilt(b,a,data)
    return y

# def notch(data, powerline, q, sr):
#     b,a = notch_filter(powerline,q, sr)
#     z = filtfilt(b,a,data)
#     return z

def butter_bandpass(lowcut, highcut, sr, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, sr, order=4):
    b, a = butter_bandpass(lowcut, highcut, sr, order=order)
    y = lfilter(b, a, data)
    return y

def final_filter(data, sr, order):
    b, a = notch_filter(60, 30, sr)
    x = filtfilt(b, a, data)     
    d, c = butter_highpass(1, sr, order = order)
    y = filtfilt(d, c, x)
    f, e = butter_lowpass(40, sr, order = order)
    z = filtfilt(f, e, y)
    return z

# diff導數: input 為 filter後的數值 與 sample rate
def diff(filter,sr):
    diffSignal=[]
    max=len(filter)-2
    const=1/(8*(1/sr))
    for index in range (2,max) :
             diffSignal.append(
                const*(
                    -1*filter[index-2]
                    -2*filter[index-1]
                    +2*filter[index+1]
                    +1*filter[index+2]
                    )
                )

    return diffSignal

# diffSignal=diff(filter)

# square平方: input 為 diffSignal的數值
def square (diffSignal):
    squaredSignal=[]

    for element in diffSignal :
        squaredSignal.append(element*element)


    return squaredSignal
# squaredSignal=square(diffSignal)

# smooth平滑: input 為 squaredSignal的數值 與 sample rate
def smooth (squaredSignal,sr):
    # shift compensate
    smoothedSignal= [0 for i in range(8)]
    add=0
    if (sr == 128):
        for index in range (10,len(squaredSignal)-11) :
            for count in range (0,11) :
                add+=(1/11 *(squaredSignal[index-count]))
            smoothedSignal.append(add)
            add=0
    elif(sr == 512 or sr ==360 or sr == 500):
        for index in range (30,len(squaredSignal)-31) :
            for count in range (0,31) :
                add+=(1/31 *(squaredSignal[index-count]))
            smoothedSignal.append(add)
            add=0

    
    return smoothedSignal

def ecg_preprocess(ecg_sig, sr):
    diffSignal=diff(ecg_sig,sr)
    filtered = final_filter(diffSignal, sr, 4)
    squaredSignal=square(filtered)
    smoothedSignal=smooth(squaredSignal,sr)
    return smoothedSignal



def rolling(data, l):
    data_avg = np.mean(data)
    result = [data_avg for i in range(len(data))]
    for i in range(l-1,len(data)):
        result[i] = np.mean(data[i-(l-1):i+1])
    return result        
    
def ecg_hr_estimate(smooth_ecg_sig,sr): # 輸入為 ECG preprocessing
   
#讀取csv檔時的配置
    data = smooth_ecg_sig 


#計算兩個方向1s的移動平均，然後加入dataset
    hrw = 1 #每個數據點兩側1秒窗口計算移動平均值

    ecg_rolling = rolling(data, int(sr*hrw))
    
    ecg_rolling = [x*2 for x in ecg_rolling] #防止移動平均線過低抓到不是R的Peak
    
# Mark regions of interest
    window = []
    peaklist = []
    error = []
    listpos = 0 #計數器

#第一次抓取檢測到的 R Peak 
    for datapoint in data:
        rollingmean = ecg_rolling[listpos] #取得平均值
        if (datapoint < rollingmean) and (len(window) < 1): #如果沒有可檢測到的 R Peak -> 什麼也不做
            listpos += 1

        elif (datapoint > rollingmean): #如果信號高於局部平均值，標記ROI
            window.append(datapoint)
            listpos += 1
            
        else: #如果信號低於局部平均值 -> 確定最高點
            beatposition = listpos - len(window) + (window.index(max(window))) # 標註點在X軸上的位置
            peaklist.append(beatposition) # 將檢測到的最大值添加到列表中R-Peak
            window = [] # 清除ROI
            listpos += 1
    ybeat = [data[x] for x in peaklist] # 獲取所有在y軸上R_Peak振幅值

    y = np.mean(ybeat)*0.3 # 平均所有R-Peak高度乘上0.3，以作為濾除較低的雜訊

    RR2 = []
    cnt2 = 0
    peak = []
#第二次 檢測 R-Peak 的正確性
    while (cnt2 < (len(peaklist)-1)):
        RR = (peaklist[cnt2+1] - peaklist[cnt2]) 
        RR2.append(RR)
        upper_threshold = abs(np.mean(RR2) + (np.mean(RR2)*0.9)) # 設定上閥值
        lower_threshold = abs(np.mean(RR2)- (np.mean(RR2)*0.3)) #設定下閥值
        if upper_threshold > RR > lower_threshold and ybeat[cnt2+1] > y : # 判斷過近相鄰點位或過遠的點位與濾除低點雜訊
            peak.append(peaklist[cnt2+1]) # 如果是放入正確Peaklist中計數器+1
            cnt2 += 1
        else:
            error.append(peaklist[cnt2]) # 如果是放入Error Peaklist中計數器+1
            cnt2 += 1

    peak.insert(0,peaklist[0])# 因為取peaklist[cnt+1]所以要加回第一個R-Peak值

    RR_list = []
    RR2 = []
    cnt3 = 0
    bpm1 = []

# #計算心率 利用第二次 檢測 的R-Peak平均值計算上下限 再計算RR interval
    while (cnt3 < (len(peak)-1)):
        RR_interval = (peak[cnt3+1] - peak[cnt3]) 
        RR2.append(RR_interval)
        upper = abs(np.mean(RR2) + (np.mean(RR2)*0.3)) #Set thresholds
        lower = abs(np.mean(RR2) - (np.mean(RR2)*0.3))
        if upper > RR_interval > lower :
            ms_dist = ((RR_interval / sr) * 1000.0) 
            RR_list.append(ms_dist) 
            cnt3 += 1
            bpm = 60000 / ms_dist
            bpm1.append(bpm)
        else :
            cnt3 += 1
 

    return bpm1, peak


