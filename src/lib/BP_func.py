import numpy as np
import os, sys
import math
import mat73
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.abspath('../lib'))
import sig_proc as sp

class feature_extraction:
    def __init__(self, sr, filt_ppg):
        self.filt_ppg = filt_ppg
        self.sr= sr
        # find peak and trough
        self.pks_loc, self.trs_loc = self.find_peak_trough()         
        
        # segment each pulse
        self.pulse_loc_set = self.pulse_seg() 
        self.num_pulse = len(self.pulse_loc_set)
        
        
        # quality inspection for each pulse
        self.tm_initial_flag = False
        self.hq_idx = np.zeros(self.num_pulse) 
        self.qi_flag = self.get_qi()
        


    def get_qi(self):
        # check numer of pulse, at lesat have 2 pulse
        if self.num_pulse > 1:
            self.hq_idx = self.quality_inspection(xcorr_thr=0.8,num_tm=3) 
            # check that sample is at least has 1 good pulse and template has initialization
            if sum(self.hq_idx) > 0 and self.tm_initial_flag:
                return True
            else:
                return False
        else:
            return  False
    
    # get ppg based and physiological features and combine as list, has 15 features
    def get_feature(self):
        if self.qi_flag:
            feat_tab1 = self.ppg_features() # 7 features
            # feat_tab2 = self.phy_features() # 8 features
            # combin feature list
            return feat_tab1
        else:
            return [None for i in range(7)]
        
    # interpolate into fix data length for template matching using
    def single_pulse_tailor(self, pulse_segment, pulse_width):
        pulse_width = int(pulse_width)

        y = np.array(pulse_segment)
        f = interp1d(np.arange(y.size),y)
        interp_ppg = f(np.linspace(0,y.size-1, pulse_width))
    
        return interp_ppg
    
    def quality_inspection(self, xcorr_thr=0.8, num_tm=3):
        
        pulse_loc = self.pulse_loc_set
        hq_idx = np.zeros(self.num_pulse)
        duration = np.zeros(self.num_pulse)
        for n in range(self.num_pulse):
            tr1_loc = pulse_loc[n][0]
            tr2_loc = pulse_loc[n][2]
            duration[n] = tr2_loc- tr1_loc
        
        # Dynamic set the length of template
        template_length = int(np.median(duration))
        max_len = 240/60 # minimum HR 240
        min_len = 40/60 # minimum HR 40
        if template_length > self.sr*max_len:
            template_length = int(self.sr*max_len)
        elif template_length < self.sr*min_len:
            template_length = int(self.sr*min_len)
        # Template Initial
        tm_set = []
        fix_pulse_set = np.zeros((self.num_pulse, template_length))
        for n in range(self.num_pulse):
            tr1_loc = pulse_loc[n][0]
            tr2_loc = pulse_loc[n][2]
            pulse_segment = self.filt_ppg[tr1_loc: tr2_loc+1]
            single_pulse= self.single_pulse_tailor(pulse_segment, template_length)
            fix_pulse_set[n] = single_pulse
            
            if n == 0:
                ref_pulse = single_pulse
            else:
                test_pulse = single_pulse
            
                # Pulse similarity inspection using cross correlation 
                xcorr = sp.corrcoef(test_pulse, ref_pulse)
                # Store the index of good quality pulse
                if xcorr > xcorr_thr:
                    tm_set.append(test_pulse)       
            
                # update previous pulse
                ref_pulse = test_pulse   
        if tm_set !=[] and len(tm_set)> num_tm:
            self.tm_initial_flag = True
            avg_template = np.mean(tm_set, axis=0)
        
        if self.tm_initial_flag:
            for n in range(self.num_pulse):
                fix_pulse = fix_pulse_set[n]
                cr_score = sp.corrcoef(avg_template, fix_pulse)
                if cr_score > xcorr_thr:
                    hq_idx[n] = 1
        
        return hq_idx 
     
    def phy_features(self):

        sampling_time = 1/self.sr # sample rate
        l = self.num_pulse
        s = self.filt_ppg
        
        v = [0.1, 0.25, 0.33, 0.5]
        ppg_st = np.zeros(len(v))
        ppg_dt = np.zeros(len(v))
        
        # take pulse 
        for n in range(l):
            if self.hq_idx[n] ==1: 
                trLoc0 = self.pulse_loc_set[n][0]
                trLoc1 = self.pulse_loc_set[n][2]
                pkLoc = self.pulse_loc_set[n][1]
                break
            
        pk = s[pkLoc]
        tr0 = s[trLoc0]
        tr1 = s[trLoc1]
        
        for j in range(len(v)):
            for i in range(trLoc0, pkLoc):
                if s[i]>=(v[j]*pk + tr0):
                    stp = i
                    break
            
            for k in range(pkLoc, trLoc1):
                if s[k] <= (v[j]*pk + tr1):
                    dtp=k
                    break
            
            ppg_st[j] = (pkLoc - stp)*sampling_time
            ppg_dt[j] = (dtp- pkLoc)*sampling_time
        
        '''
        ['Sw10', 'St10+Dt10','Dt10/St10', 'St25+Dt25','Dt25/St25',
            'St33+Dt33', 'Dt33/St33','Dt50/St50'] 
        '''
        return [ppg_st[0], ppg_st[0]+ppg_dt[0], ppg_dt[0]/ppg_st[0],
                ppg_st[1]+ppg_dt[1], ppg_dt[1]/ppg_st[1],
                ppg_st[2]+ppg_dt[2], ppg_dt[2]/ppg_st[2],
                ppg_dt[3]/ppg_st[3]]
        
    
    # 7 PPG based feature 
    def ppg_features(self):
    
        sr = self.sr # sample rate
        l = self.num_pulse
        HR=0 # Heart rate
        St=0 # Systolic time
        Dt=0 # Diastolic time
        Ih = 0 # max amplitude of pulse signal
        Il = 0 # min amplitude of pulse signal
        PIR = 0 # PPG Intensity Ratio, ratio of max and min amplitude of pulse signal
        alpha = 0 # Wormersley number
        cnt=0
        for i in range(l):
            if self.hq_idx[i]==1:
                trLoc0 = self.pulse_loc_set[i][0]
                trLoc1 = self.pulse_loc_set[i][2]
                pkLoc = self.pulse_loc_set[i][1]
                SDI = (trLoc1-trLoc0)/sr # Systolic Diastolic time interval
                tr_amp = (self.filt_ppg[trLoc1] + self.filt_ppg[trLoc0])*0.5
                pk_amp = self.filt_ppg[pkLoc]
                
                HR_tmp = (sr/(trLoc1-trLoc0))*60
                    
                HR += HR_tmp
                alpha += tr_amp*math.sqrt(1060*HR_tmp/pk_amp)
                    
                St += (pkLoc-trLoc0)/sr
                Dt += (trLoc1-pkLoc)/sr
                Ih += pk_amp
                Il += tr_amp
                cnt+=1
        if cnt != 0:
            HR = HR/cnt
            alpha = alpha/cnt
            St = St/cnt
            Dt = Dt/cnt
            Ih = Ih/cnt
            Il = Il/cnt
            PIR = Ih/Il
        
        return [HR, alpha, St, Dt, Ih, Il, PIR]

   
        
    # Feed ppg peak and its valley locations,
    # return 2d array [[trough_loc0, peak_loc, trough_loc1], [], ...]
    def pulse_seg(self):
        ppg_peak_loc = self.pks_loc
        ppg_trough_loc = self.trs_loc
        pulse_loc_set = []
        for j in range(1, len(ppg_trough_loc)):
            # find single periodic wave
            trLoc0 = ppg_trough_loc[j-1]
            trLoc1 = ppg_trough_loc[j]

            # extract single pulse
            for k, loc in enumerate(ppg_peak_loc):
                if loc > trLoc0 and loc < trLoc1:
                    pkLoc = ppg_peak_loc[k]
                    pulse_loc_set.append([trLoc0, pkLoc, trLoc1])
        return pulse_loc_set
            
    def find_peak_trough(self):
        #find-peak
        sample_rate = self.sr
        filtered_ppg = self.filt_ppg
        medium_max_count = 0
        medium_min_count = 0
        peak_count =0
        valley_count = 0
        ppg_peak_x =[]
        ppg_trough_x =[]
        
        adaptive_windows_size_min = math.ceil(sample_rate*0.25) 
        adaptive_windows_size_max = math.ceil(sample_rate*0.5) #最大時窗一半
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
                            ppg_trough_x.append(i)
                            valley_count+=1                                                     
            medium_max_count = 0
            medium_min_count = 0
        return  ppg_peak_x, ppg_trough_x