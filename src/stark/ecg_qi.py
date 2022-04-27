import numpy as np
   
def q_check(sr, rpeak, filt_ecg):
    is_good = 1
    good_rr = []
    xcorr_list = np.zeros(len(rpeak))
    
    # At least have 3 peak in 4 seconds
    if len(rpeak) >2:
        rr_list = (np.diff(rpeak)/sr) # in seconds
    else:
        is_good = 0
    
    # Make sure the rr interval in normal range
    # HR between 40~180
    if is_good:
        for rr in rr_list:
            if rr > 1.5 or rr < 0.33:
                is_good =0
                break
                
    if is_good == 1:
        max_rr = max(rr_list)
        min_rr = min(rr_list)
        rr_ratio = max_rr/min_rr
        if rr_ratio > 2.2:
            is_good = 0
            
        # similarity
        else:
            rr_med_lenght = np.median(rr) * sr
            for i in range(1, len(rpeak)):
                
                # left and right bound of current samples
                lb = rpeak[i] - int((rr_med_lenght/2) + 0.5)
                rb = rpeak[i] + int((rr_med_lenght/2) + 0.5)
                
                # left and right bound of reference samples
                ref_lb = rpeak[i - 1] - int((rr_med_lenght/2) + 0.5)
                ref_rb = rpeak[i - 1] + int((rr_med_lenght/2) + 0.5)
                
                # make sure not out of the sample data
                if ref_lb > 0 and rb < len(filt_ecg):
                    
                    current_sample = filt_ecg[lb:rb]
                    ref_sample = filt_ecg[ref_lb:ref_rb]
                    
                    
                    xcorr = np.round(corrcoef(current_sample, ref_sample),2)
                    xcorr_list[i] = xcorr

                    # cross correlation threshold
                    if xcorr > 0.7:
                        good_rr.append((rpeak[i] - rpeak[i-1])/sr)
    if good_rr ==[]:
        is_good = 0
    return is_good
            
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