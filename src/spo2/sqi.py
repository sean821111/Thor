

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math
import sys


sys.path.insert(0, os.path.abspath('../lib'))
import sig_proc as sp


# currently for 4 seconds version
def signalQuality(filt_R, filt_IR, sr):
    tm_length=2*sr
    template_R = filt_R[0:tm_length]
    template_IR = filt_IR[0:tm_length]
    test_signal_R = filt_R[tm_length:]
    test_signal_IR = filt_IR[tm_length:]

    r_xcorr = sqi_xcorr(test_signal_R, template_R)
    ir_xcorr = sqi_xcorr(test_signal_IR, template_IR)
    ri_corr = sp.corrcoef(test_signal_R, test_signal_IR)

    if r_xcorr <0.5 or ir_xcorr<0.5 or ri_corr < 0.7:
        level = 'low'
    elif (r_xcorr > 0.5 and r_xcorr <0.7) or (ir_xcorr > 0.5 and ir_xcorr<0.7):
        level = 'medium'
    else:
        level='high'
        
    return level




def sqi_xcorr(x,y):
    x_norm = sp.z_score(x)
    y_norm = sp.z_score(y)
    pxy = max(np.correlate(x_norm, y_norm))
#     pyy = np.correlate(y_norm, y_norm)
    pyy = sum(y_norm*y_norm)
    return (2*pxy)/pyy

