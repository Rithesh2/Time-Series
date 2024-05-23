# for plotting, run: pip install pandas matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.signal import medfilt, sosfilt, ellip, iirnotch, freqz, filtfilt
from sklearn.preprocessing import MinMaxScaler

import wfdb
# !pip install gluonts

# from gluonts.dataset.pandas import PandasDataset
# from chronos_mlx import ChronosPipeline

# pipeline = ChronosPipeline.from_pretrained(
#     "amazon/chronos-t5-small",
#     dtype="bfloat16",
# )

def create_record(filepath, column_subset = ['MLII']):
    record = wfdb.rdrecord(filepath)
    df = record.to_dataframe()[column_subset]
    return df


def quantile_outlier_removal(df:pd.DataFrame, column_name:str, quantile_min:float, quantile_max:float, plot:bool = False):
    minmax_quantiles = np.quantile(df[column_name], [0.0001,0.9999])

    df_removed = df[((df>minmax_quantiles[0]) & (df<minmax_quantiles[1]))].ffill().bfill()
    counts, values = np.histogram(df, bins = 100)
    
    if plot:
        (df).hist(bins = 100)
        plt.plot(np.full(2, minmax_quantiles[0]), np.linspace(0, np.max(counts), 2), linewidth = 2, color = 'k', linestyle = '--', label = 'min value cutoff')
        plt.plot(np.full(2, minmax_quantiles[1]), np.linspace(0, np.max(counts), 2), linewidth = 2, color = 'k', linestyle = ':', label = 'max value cutoff')
        plt.title('Quantile Removal')
        plt.legend()
        plt.show()


    return df_removed

def simple_moving_average(df:pd.DataFrame, window:int, plot = False):
    df_sma = df.rolling(window).mean()
    if plot:
        plt.plot(df_sma, linewidth = 0.1)
        plt.title(f'Simple Moving Average ({window}-frame)')
        plt.show()
    return df_sma

def median_filtering(df:pd.DataFrame, window:int, plot = False):
    df_medfilt = medfilt((df).values.flatten(),3)
    df_medfilt = pd.Series(df_medfilt).bfill()
    if plot:
        plt.plot(df_medfilt, linewidth = 0.1)
        plt.title(f'Median Filtering ({window}-frame)')
        plt.show()
    return df_medfilt

def sos_filter(df:pd.DataFrame, plot = False, N=2, rp=0.09, rs=80, Wn=0.09, output = 'sos'):
    sos = ellip(N = N, rp = rp, rs = rs, Wn = Wn, output=output)
    df_sos = sosfilt(sos, df)
    if plot:
        plt.plot(df_sos, linewidth = 0.1)
        plt.title('Second Order Section Filter')
    return df_sos

def notch_filtering(df:pd.DataFrame, plot = False, fs = 360.0, f0 = 60, Q = 30):

# Remove 60Hz analog-digital conversion tone from signal

# fs - Sample frequency (Hz)

# f0 - Frequency to be removed from signal (Hz)

# Q - Quality factor

# Design notch filter

    b_notch, a_notch = iirnotch(f0, Q, fs)

    df_notch = filtfilt(b_notch, a_notch, df)
    if plot:
        plt.plot(df_notch, linewidth = 0.1)
        plt.title('Notch Filter')
        plt.show()
    return df_notch

def normalizer(df:pd.DataFrame, scaler = MinMaxScaler(), plot = False):

    df_scaled = scaler.fit_transform(df.reshape((-1,1))).flatten()
    if plot:
        plt.plot(df_scaled, linewidth = 0.1)
        plt.title('Scaled')
        plt.show()
    return df_scaled

