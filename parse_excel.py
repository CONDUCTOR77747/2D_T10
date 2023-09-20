# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:57:42 2023

@author: ammosov_yam
"""

import pandas as pd
import re

DAS_FILES_path = 'C://Progs//DAS_FILES//'
# path = "C://Progs//Py//2D_t10_documents//T10_B22_I230//2d_t10_QCM//Shots_TI_FI.xlsx"
pathe = "C://YandexDisk//Компьютер AMMOSOVYM//Py//2D_t10_documents//T10_B22_I230//2d_t10_QCM//Shots_TI_FI.xlsx"

def load_ebeam(shot, path):
    try:
        with open(path + 'n' + str(shot) + '.inf', 'r') as f:
            lines = f.readlines()
        f.close()
        return round(float((re.findall(r'\d+[.]+\d*', lines[1]))[0]))
    except IOError:
        print('\nERROR: Invalid DAS_FILES location\n')

def create_df_from_excel(path, sheet_name):
    df = pd.DataFrame()
    df = pd.read_excel(path, sheet_name='OH')
    df = df.where(pd.notnull(df), None)
    return df

def parse_interval(strin):
    a, b = re.findall(r'\d+[.]\d+', strin)
    return "{:.2f}".format(float(a)), "{:.2f}".format(float(b))

def make_interval(path):
    a, b = parse_interval(path)
    return f"from{a}to{b}"


if __name__ == '__main__':
    
    df = create_df_from_excel(pathe, "OH")
    
    for shot in df:
        
        columnSeriesObj = df[shot]
        columnSeriesObjvalues = columnSeriesObj.values
        
        for val in columnSeriesObjvalues:
            if val:
                interval = make_interval(val)
                ebeam = load_ebeam(shot, DAS_FILES_path)
                strin = f"{shot} !{ebeam} !{interval}"
                print(strin)

