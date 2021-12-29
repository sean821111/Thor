import numpy as np
import matplotlib.pyplot as plt
import os, sys
import mat73
import csv
from time import sleep
import progressbar
import math

sys.path.insert(0, os.path.abspath('../lib'))
import BP_func 

plt.style.use('ggplot')


class blood_pressure(BP_func.feature_extraction):
    def __init__(self, sr, abp):
        self.sr=sr
        self.filt_ppg = abp



def get_bp(value):
    if not isinstance(value, blood_pressure):
        raise ValueError("value muse be BP instance")
    pk_locs, tr_locs = value.find_peak_trough()
    sbp = np.mean(value.filt_ppg[pk_locs])
    dbp = np.mean(value.filt_ppg[tr_locs])
    return sbp, dbp

if __name__ == '__main__':
    folder = '../../../MIMIC II/'
    if len(sys.argv) < 2:
        data_name = 'Part_1'
    else:
        data_name = sys.argv[1]

    sr = 125
    # Load data
    folder = '../../../MIMIC II/'
    data_path = folder  + data_name
    print('Processing data path: ', data_path)
    data_dict = mat73.loadmat(data_path + '.mat')
    data_len = 3000
    
    # progress bar
    bar = progressbar.ProgressBar(maxval=data_len, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # rows = np.zeros((data_len, 10))
    rows = []
    for i in range(data_len):
        
        ppg = data_dict[data_name][i][0][0:1000]
        abp = data_dict[data_name][i][1][0:1000]
        # ecg = data_dict['Part_1'][i][2][0:1000]
        
        pf = BP_func.feature_extraction(sr, ppg)
        sbp, dbp = get_bp(blood_pressure(sr, abp))
        
        feat_tab = pf.get_feature()
        
        row = []
        if feat_tab[0] != None and not math.isnan(sbp) and not math.isnan(dbp):
            row.append(int(i))
            row.append(sbp)
            row.append(dbp)
        
            for f in feat_tab:
                row.append(f)
            rows.append(row)
        
        bar.update(i+1)
        sleep(0.1)

    bar.finish()

    # open the file in the write mode
    # writing to csv file 
    with open(data_name + '_feature.csv', 'w', newline="") as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the data rows 
        csvwriter.writerows(rows)
