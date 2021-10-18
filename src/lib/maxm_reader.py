import os
import re
import csv


# For read maxim excel file

'''
folder: contain all subject
subfolder: specific subject
source: using regular expression check file name 
'''
def maxm_reader(folder, subfolder, source="MAXM"):
    file_path = os.path.join(folder, subfolder)
    for f in os.listdir(file_path):
        if re.findall(source, f):
            ext = os.path.splitext(str(f))[-1].lower()
            if ext == '.csv':
                file_name = str(f)
                f_check = True
                break
            else:
                f_check = False

    if f_check:
        full_path = os.path.join(file_path, file_name)
        reader = csv.reader(open(full_path))

        dataDict = []
        for i, row in enumerate(reader):
            if i == 6:
                first_row = row[:-1] #ignore last empty value
                dataDict = {r:[] for r in first_row}
            elif i > 6:
                if row[0] == 'stop time':
                    break
                for col in range(len(first_row)):
                    if row[col] == '':
                        _value = None
                    else:
                        _value = float(row[col])
                    dataDict[first_row[col]].append(_value)
        return dataDict
    else:
        raise Exception('File not found:{}'.format(file_path))