import numpy as np
import matplotlib.pyplot as plt
import os,sys
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.abspath('lib'))
import sig_proc as sp

#　calculate rr interval for ppg signal 
def preprocess(sr, ppg_signal):
    pks_loc, _ = sp.find_peak_valley(sr, ppg_signal)
    rr = (np.diff(pks_loc)/sr)*1000
    return rr


def timedomain(rr):
    results={}
    rr = np.array(rr)
    hr = 60000/rr
    
    # results['Mean RR (ms)'] = np.mean(rr)
    results['Mean HR (beats/min)'] = np.mean(hr)
    # results['STD HR (beats/min)'] = np.std(hr)
    # results['Min HR (beats/min)'] = np.min(hr)
    # results['Max HR (beats/min)'] = np.max(hr)
    results['SDNN (ms)'] = np.std(rr)
    results['RMSSD (ms)'] = np.sqrt(np.mean(np.square(np.diff(rr))))
    results['NN50'] = np.sum(np.abs(np.diff(rr)) > 50)*1
    results['pNN50 (%)'] = 100 * np.sum((np.abs(np.diff(rr)) > 50)*1) / (len(rr)-1)
    return results

def interp(rr):
    fs = 4
    # create interpolation function based on the rr-samples. 
    t = np.cumsum(rr)
    t -= t[0]
    f_interpol = CubicSpline(t, rr, bc_type='natural')
    # f_interpol = interp1d(t, rr, 'cubic')
    t_interpol = np.arange(t[0], t[-1], 1000./fs)
    rr_interpol = f_interpol(t_interpol)
    
    rr_interpol = rr_interpol - np.mean(rr_interpol)
    
    return rr_interpol
    
    
# Due to the edge device memory issue, change the n_fft size from 4096 to 512
def freqdomain(rr, nfft=2**9):

    # perform welch's method 
    # fxx, pxx = signal.welch(x=rr_interpolated, fs=fs, nperseg=nperseg,nfft=2**12, scaling='density')
    fxx, pxx = welch_psd(rr, nfft=nfft, fs=4.0)
    
    '''
    Segement found frequencies in the bands 
    - Very Low Frequency (VLF): 0-0.04Hz 
    - Low Frequency (LF): 0.04-0.15Hz 
    - High Frequency (HF): 0.15-0.4Hz
    '''
    cond_vlf = (fxx >= 0) & (fxx < 0.04)
    cond_lf = (fxx >= 0.04) & (fxx < 0.15)
    cond_hf = (fxx >= 0.15) & (fxx < 0.4)
    
    # calculate power in each band by integrating the spectral density 
    # vlf = trapz(pxx[cond_vlf], fxx[cond_vlf])
    # lf = trapz(pxx[cond_lf], fxx[cond_lf])
    # hf = trapz(pxx[cond_hf], fxx[cond_hf])
    df = fxx[1]-fxx[0]
    vlf = np.sum(pxx[cond_vlf]) * df
    lf = np.sum(pxx[cond_lf]) * df
    hf = np.sum(pxx[cond_hf]) * df
    
    # sum these up to get total power
    total_power = vlf + lf + hf

    # find which frequency has the most power in each band
    peak_vlf = fxx[cond_vlf][np.argmax(pxx[cond_vlf])]
    peak_lf = fxx[cond_lf][np.argmax(pxx[cond_lf])]
    peak_hf = fxx[cond_hf][np.argmax(pxx[cond_hf])]

    # relative power
    vlf_nu = 100 * vlf / total_power 
    lf_nu = 100 * lf / total_power
    hf_nu = 100 * hf / total_power
    
    results = {}
    results['Power VLF (ms2)'] = vlf
    results['Power LF (ms2)'] = lf
    results['Power HF (ms2)'] = hf   
    results['Power Total (ms2)'] = total_power

    results['LF/HF'] = (lf/hf)
    results['Peak VLF (Hz)'] = peak_vlf
    results['Peak LF (Hz)'] = peak_lf
    results['Peak HF (Hz)'] = peak_hf
    
    results['Relative VLF (nu)'] = vlf_nu
    results['Relative LF (nu)'] = lf_nu
    results['Relative HF (nu)'] = hf_nu
    return results, fxx, pxx


# According to pyHRV, a python toolbox for HRV measurement
# https://github.com/PGomes92/pyhrv/blob/593695632f0505cc29d989fd6cf2a24a07166947/pyhrv/README.md

def welch_psd(rr, nfft=2**12, fs=4):
    t = np.cumsum(rr)
    t -= t[0]
    f_interpol = CubicSpline(t, rr, bc_type='natural')
    # f_interpol = interp1d(t, rr, 'cubic')
    t_interpol = np.arange(t[0], t[-1], 1000./fs)
    rr_interpol = f_interpol(t_interpol)
    
    rr_interpol = rr_interpol - np.mean(rr_interpol)
    
    if t.max() < 300000:
        nperseg = nfft
    else:
        nperseg = 300
    
    # Self define, independent library 
    freqs, powers = csd(x=rr_interpol, fs=4.0, nperseg=nperseg, nfft=nfft)
    
    # scipy signal welch's method
    # freqs, powers = signal.welch(
	# 	x=rr_interpol,
	# 	fs=fs,
	# 	window='hamming',
	# 	nperseg=nperseg,
	# 	nfft=nfft,
	# 	scaling='density'
	# )
    return freqs, powers


def csd(x, fs=4.0, nperseg=None, noverlap=None, nfft=None):
    
    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')
    # parse window; if array like, then set nperseg = win.shape
    window='hamming'
    # win, nperseg = sss._triage_segments(window, nperseg, input_length=x.shape[-1])
    win, nperseg = _triage_segments(nperseg, input_length=x.shape[-1])
    
    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)
        
    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    
    scale = 1.0 / (fs * (win*win).sum())
    
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    # freqs = sp_fft.rfftfreq(nfft, 1/fs)
    
    # Perform the windowed FFTs
    # result = sss._fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides='onesided')    
    result = _fft_helper(x, win, nperseg, nfft)    
    # result = np.fft.rfft(result_own, 2**12)
    result = np.conjugate(result) * result
    result *= scale

    if nfft % 2:
        result[1:] *= 2
    else:
        # Last point is unpaired Nyquist freq point, don't double
        result[1:-1] *= 2
    return freqs, result

def _fft_helper(x, win, nperseg, nfft):
    if nperseg < x.shape[-1]:
        result = x[:nperseg]
    else:
        result = x
        
    # detrend function
    result = result - np.mean(result)

    # Apply window by multiplication
    result = win * result

    result = np.fft.rfft(result, nfft)
    return result

def _triage_segments(nperseg, input_length):
    
    if nperseg > input_length:
        nperseg = input_length
        
    
    win = hamming_window(nperseg, [0.54, 1.-0.54])
    
    # for n in range(nperseg):
        # win[n] = 0.54-(0.46*np.cos((2*np.pi*n)/(nperseg-1)))
    # win = sss.get_window(window, nperseg)

    return win, nperseg

def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1

def hamming_window(M, a):

    if _len_guards(M):
        return np.ones(M)
    # M, needs_trunc = _extend(M, sym)

    fac = np.linspace(-np.pi, np.pi, M)
    w = np.zeros(M)
    for k in range(len(a)):
        w += a[k] * np.cos(k * fac)

    return w
