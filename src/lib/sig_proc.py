import numpy as np 
from scipy.sparse import csc_matrix,spdiags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
import math
import csv
from statistics import mean, stdev,median
import scipy.signal as signal
from scipy.interpolate import interp1d


def value_trans(data):
    data = data*1.2/2097151.0
    return data

'''
上下翻轉
:param s: float array
:return ppg_flip: float array
'''
def flip_up_down(s)->list:
    _maximum = max(s)

    ppg_flip = []
    for value in s:
        ppg_flip.append(_maximum- value)
        
    return ppg_flip

def avg_filter(signal, filter_size):
    h = np.ones(filter_size)/filter_size
    y = np.convolve(signal, h, 'same')
    return y 


'''
SNR calculation

:param filtered_signal: float array
:param raw_signal: float array

:return SNR: float
'''
def snr_calc(filtered_signal, raw_signal):
    norm_y = z_score(filtered_signal) # filtered signal
    norm_x = z_score(raw_signal) # raw signal 
    SD_s = stdev(abs(np.array(norm_y))) **2
    SD_n = stdev(abs(np.array(norm_x))) **2
    SNR = round(SD_s/ SD_n,4)
    return SNR

'''
Z-score standardization

:param x: float array
:return : float array
'''
def z_score(x):
    return (x - np.mean(x)) / np.std(x, ddof = 1)

def normalized(a, maximum=None):
    if maximum == None:
        maximum = max(a) 
    return [value/maximum for value in a]

''' return filter parameters b,a'''
def bp_filter(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    return b, a

def hp_filter(cutoff, fs, order):
    nyq = 0.5 * fs
    cut = cutoff/ nyq
    b, a = signal.butter(order, cut, btype='highpass', analog=False)
    return b,a

def lp_filter(cutoff, fs, order):
    nyq = 0.5 * fs
    cut = cutoff/ nyq
    b, a = signal.butter(order, cut, btype='low', analog=False)
    return b,a


''' return filtered signal '''
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    cut = cutoff/ nyq
    b, a = signal.butter(order, cut, btype='highpass', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    cut = cutoff/ nyq
    b, a = signal.butter(order, cut, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

'''
ppg feature extraction

:param sample_rate: int
:param filtered_ppg:float array
:param ppg_time_list: int array
:param scaling_factor:

:return ppg_peak_x: int array
:return ppg_peak_time_x: int array
:return ppg_valley_x: int array
:return ppg_valley_time_x: int array
:return adaptive_windows_size: int 
'''
def find_peak_valley(sample_rate, filtered_ppg):
      #find-peak
    medium_max_count = 0
    medium_min_count = 0
    peak_count =0
    valley_count = 0
    ppg_peak_x =[] # peak location index
    ppg_valley_x =[] # valley location index
    adaptive_windows_size_min = math.ceil(sample_rate*0.25/2)  #最小時窗一半
    adaptive_windows_size_max = math.ceil(sample_rate/2) #最大時窗一半
    adaptive_windows_size = adaptive_windows_size_min #初始設定windows大小
    for i in  range(len(filtered_ppg)):
        if ((i > adaptive_windows_size) and (i <len(filtered_ppg)-adaptive_windows_size)):
            if (adaptive_windows_size > adaptive_windows_size_max):
                adaptive_windows_size = adaptive_windows_size_max   
            #找峰值    
            for j in range(math.ceil(adaptive_windows_size +1)):
                if ((filtered_ppg[i] > filtered_ppg[i -j])  and (filtered_ppg[i] >= filtered_ppg[i + j]) ):
                    medium_max_count = medium_max_count + 1
                    #中間大於左右次數
                    if (medium_max_count == math.ceil(adaptive_windows_size)):
                        
                        #如 a = adaptive_windows_size 可知 ppg_data(x,1)為最大值
                        ppg_peak_x.append(i)

                        if (len(ppg_peak_x) > 1 ): #更新視窗寬度
                            adaptive_windows_size = math.ceil ((ppg_peak_x [peak_count] - ppg_peak_x [peak_count-1]) / 2) ;
                            
                        peak_count+=1 
                elif ((filtered_ppg[i] < filtered_ppg[i -j])  and (filtered_ppg[i] <= filtered_ppg[i + j]) and len(ppg_peak_x)>0 ):
                    medium_min_count = medium_min_count +1
                    #中間小於左右次數
                    if (medium_min_count == math.ceil(adaptive_windows_size)):
                        ppg_valley_x.append(i)
                        valley_count+=1                                                     
        medium_max_count = 0
        medium_min_count = 0

    return  ppg_peak_x, ppg_valley_x


# Feed ppg peak and its valley locations, return it as tulple format
# return 2d array [[trough_loc0, peak_loc, trough_loc1], [], ...]
def pulse_seg(ppg_peak_loc, ppg_valley_loc):
    pulse_loc_set = []
    for j in range(1, len(ppg_valley_loc)):
        # find single periodic wave
        v0Loc = ppg_valley_loc[j-1]
        v1Loc = ppg_valley_loc[j]

        # extract single pulse
        for k, loc in enumerate(ppg_peak_loc):
            if loc > v0Loc and loc < v1Loc:
                pkLoc = ppg_peak_loc[k]
                pulse_loc_set.append([v0Loc, pkLoc, v1Loc])
                # avoid two peak
                break
    return pulse_loc_set

# interpolate into fix data length for template matching 
def single_pulse_tailor(pulse_loc, filtered_ppg, pulse_width=40):
    pulse_width = int(pulse_width)
    v0Loc = pulse_loc[0]
    # pkLoc = pulse_loc[1]
    v1Loc = pulse_loc[2]
    y = np.array(filtered_ppg[v0Loc:v1Loc])
    f = interp1d(np.arange(y.size),y)
    interp_ppg = f(np.linspace(0,y.size-1, pulse_width))
   
    return interp_ppg

# calculate single PPG pulse area 
def pulse_area(x1, x2, curve):
    pulse = np.array(curve[x1:x2])
    pulse_len = len(pulse)
    m = (pulse[pulse_len-1] - pulse[0])/pulse_len
    # y = mx+ b
    b = pulse[0]
    # y_list=[]
    area = 0
    for x in range(pulse_len):
        #y_list.append(m*x + b)
        y = m*x+b
        area += pulse[x] - y
    
    return area


'''
Parameters:
    file_path: string
        full file path

    source: string
        e.g. G2, G1, _R, IR, ACC
'''
def ppg_preprocess(ppg_data, sr, norm=1, flip=1):
    # band pass filter parameters
    b_b, b_a = bp_filter(0.5, 10, sr, 4)
    
    if flip:
        ppg_flip = flip_up_down(ppg_data)
    else:
        ppg_flip = ppg_data

    # band pass filter
    filtered_ppg = signal.filtfilt(b_b, b_a, ppg_flip)
    # filtered_ppg= move_avg(filtered_ppg, 5)
    if norm:
        filtered_ppg = z_score(filtered_ppg)
    return filtered_ppg


'''
x: int array
y: int array
'''
def corrcoef(x,y):
    n = len(x)
    x_mu = np.mean(x)
    y_mu = np.mean(y)
    x_sigma = np.std(x)
    y_sigma = np.std(y)
    
    zx = (np.array(x)-x_mu)/x_sigma
    zy = (np.array(y)-y_mu)/y_sigma

    cc = np.mean(zx*zy)
    return cc


'''
Description:
    Tailor single ppg pulse into specific width using interpolation for template matching.
Parameters:
    pulse_loc: ppg pulse location, where indicate left valley, peak and right valley respectively.
    filtered_ppg: filtered ppg list
    pulse_width: specific length for template matching
Return: 
    interp_ppg: interpolated single ppg pulse
'''
def single_pulse_tailor(pulse_loc, filtered_ppg, pulse_width=40):
    pulse_width = int(pulse_width)
    v_t0 = pulse_loc[0]
    p_t1 = pulse_loc[1]
    v_t1 = pulse_loc[2]
    y = np.array(filtered_ppg[v_t0:v_t1])
    f = interp1d(np.arange(y.size),y)
    interp_ppg = f(np.linspace(0,y.size-1, pulse_width))
   
    return interp_ppg


'''
Not adopt in current version
average filter

:param data: float array
:param win_size: int
:return avg: float array
'''
def move_avg(data, win_size):
    avg = [0 for i in range(len(data))]
    h = np.ones(win_size)/win_size
    # ignore the boundary value
    for i in range(len(data)):
        if i < int(win_size/2):
            avg[i] = data[i]
        elif i >= len(data) - int(win_size/2):
            avg[i] = data[i]
        else:
            avg[i] = sum(data[i-int(win_size/2):i + int(win_size/2) + 1]*h)
    return np.array(avg)


'''
Not use currently!

There are two parameters: p for asymmetry and λ for smoothness. Both have to be
tuned to the data at hand. We found that generally 0.001 ≤ p ≤ 0.1 is a good choice
(for a signal with positive peaks) and 10^2 ≤ λ ≤ 10^9
, but exceptions may occur. In any case one should 
vary λ on a grid that is approximately linear for log λ. Often visual
inspection is sufficient to get good parameter values

lam = 100
p = 0.001

'''
def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

