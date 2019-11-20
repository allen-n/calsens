# Utilities
import os
import pdb
import glob

# Numerical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing / Filtering
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import normalize


class OxData:
    def __init__(self, csv_path):
        self.path = csv_path
        self.data = None

    def data_file_list(self):
        files = []
        for fpath in sorted(glob.glob(self.path)):
            files.append(os.path.split(fpath)[-1])
        return files

    ### Creating data matrix ###
    def get_data_matrix(self, **kwargs):
        '''
        @args:
            path: kwargs:
            'sigma': 1, 'smoothing': 'gaussian',
                'plot': False, 'reparse': False,
                'fft_size': 1024, 'hold-out':[]
        @return:
            X - np array of all processed and normalized data
        '''
        args = {'sigma': 1, 'smoothing': 'gaussian',
                'plot': False, 'reparse': False,
                'fft_size': 1024, 'hold_out': []}

        for key, value in kwargs.items():
            if key in args:
                args[key] = value
        if self.data is not None and not args['reparse']:
            return self.data

        hold_outs = set(args['hold_out'])
        data_candidates = []

        for fpath in sorted(glob.glob(self.path)):
            fname = os.path.split(fpath)[-1]
            if fname in hold_outs:
                continue  # don't add this to the training matrix
            temp_data = self.csv_to_data(fpath, **kwargs)
            data_candidates.append(temp_data)

        self.data = np.concatenate(data_candidates)
        self.data = normalize(self.data)
        return self.data

    def csv_to_data(self, fpath, **kwargs):
        '''
        convert the passed file path fpath to a data matrix X
        '''
        args = {'sigma': 1, 'smoothing': 'gaussian',
                'plot': False, 'reparse': False,
                'fft_size': 1024, 'normalize':False}

        for key, value in kwargs.items():
            if key in args:
                args[key] = value

        sp02, pulse, start_time = self.day_to_df(fpath)
        # Preprocess for FFT
        sp02_wave = self.preproc_df_fft(
            sp02, smoothing=args['smoothing'], sigma=args['sigma'], plot=args['plot'])
        pulse_wave = self.preproc_df_fft(
            pulse, smoothing=args['smoothing'], sigma=args['sigma'], plot=args['plot'])
        # Perform FFT
        sp02_fft = self.gen_fft(sp02_wave, args['fft_size'])
        pulse_fft = self.gen_fft(pulse_wave, args['fft_size'])

        # Can separate magnitudes and angles as follows:
        # sp02_mag, sp02_angle = sp02_fft[:,:sp02_fft.sp02_fft[1]//2], sp02_fft[:,sp02_fft.shape[1]//2:]
        # pulse_mag, pulse_angle = pulse_fft[:,:pulse_fft.pulse_fft[1]//2], pulse_fft[:,pulse_fft.shape[1]//2:]

        # Normalize results and combine to final data matrix
        temp_data = np.concatenate((sp02_fft, pulse_fft), axis=1)
        if args['normalize']:
            temp_data = normalize(temp_data)

        return temp_data
    ### Data Loading ###

    def day_to_df(self, path):
        '''
        @args:
            path: os.path object representing path to OSCAR csv file to parse
        @return:
            return: tuple(sp02, pulse, start_time),
            sp02 = pd.DataFrame of SP02 data
            pulse = pd.DataFrame of Pulse data
            start_time = UTC start time in s
        '''
        df = pd.read_csv(path)
        time = df['Session'][0]
        df = df.drop(columns='Session')
        df['DateTime'] -= time  # All times start from 0 for session

        sp02 = df[df['Event'] == 'SPO2']
        sp02 = sp02.drop(columns='Event')

        pulse = df[df['Event'] == 'Pulse']
        pulse = pulse.drop(columns='Event')

        pulse = pulse.reset_index(drop=True)
        sp02 = sp02.reset_index(drop=True)
        del df
        return sp02, pulse, time

    # Waveform Manipulations

    def impute_wave(self, wave):
        '''
        @info:
            impute missing values in waveform
        @args:
            wave: np array with floats corresponding to waveform
        @return:
            None, modifies waveform array in place
        '''
        n = wave.shape[0]
        j = 0
        step = None
        for i in range(n-1):
            if i >= j and wave[i+1] == 0:
                j = i+1
                while j < n and wave[j] == 0:
                    j += 1
                if j < n:
                    step = float(wave[j] - wave[i]) / float(j-i+1)
            elif wave[i] == 0:
                wave[i] = wave[i-1]+step

    ### Waveform Manipulations ###

    def moving_avg(self, wave, N):
        '''
        @info:
            run moving average filter kernal of size N over the wave
        @params:
            wave: np array if size > N
            n: int, the size of the filter kernal
        @return:
            nothing, modifies wave in place
        '''
        n = wave.shape[0]
        if N > n or N % 2 != 1:
            print("Filter kernal needs to be odd sized.")
            return
        running_sum = np.sum(wave[:N])
        N2 = N//2
        for i in range(N2, n - N2 - 1):
            wave[i] = float(running_sum) / float(N)
            running_sum -= wave[i-N2]
            running_sum += wave[i+N2+1]
        # cumsum = np.cumsum(np.insert(wave, 0, 0))
        # return (cumsum[N:] - cumsum[:-N]) / float(N)

    def preproc_df_fft(self, df, **kwargs):
        '''
        Pre-process the given dataframe df (obtained using the day_to_df function)
        to space all points into time buckets so FFT output is meaningful. Then perform
        imputation to create the envalope of the waveform, and run a amoving average filter
        kernel of size avg_size on it.
        plot=False, sigma=1, smoothing='gaussian'):

        **kwargs: 
            plot=True/False -- plots garph (default False)
            sigma=int() -- smoothness of gaussian filter (default 1), moving avg size of moving avg kernel
            smoothing='gaussian'/'avg' -- whether to use gaussian or moving avg filter (default gassian)
        '''
        args = {'sigma': 1, 'smoothing': 'gaussian', 'plot': False}
        for key, value in kwargs.items():
            if key in args:
                args[key] = value

        result = np.zeros(np.max(df['DateTime'])+1)
        # TODO: fix this, DateTimes come in duplicate pairs
        result[df['DateTime']] = df['Data/Duration']

        if args['plot']:
            plt.clf()
            wave = np.array(result, copy=True)
            self.impute_wave(wave)
            if args['smoothing'] == 'gaussian':
                wave = gaussian_filter1d(wave, args['sigma'])
            else:
                self.moving_avg(wave, args['sigma'])
            plt.clf()
            plt.plot(np.arange(wave.shape[0]), result)
            plt.plot(np.arange(wave.shape[0]), wave, 'r')
            plt.xlim(20, 1000)
            plt.ylim(85, 105)
            return wave
        else:
            self.impute_wave(result)
            if args['smoothing'] == 'gaussian':
                result = gaussian_filter1d(result, args['sigma'])
            else:
                self.moving_avg(result, args['sigma'])
            return result

    ### Generation of feature matrix from day-data ###
    # TODO: Turn this function into Gen-Features, fator FFT part out, use it to generate feature array X

    def gen_fft(self, wave, kernel_size=256, isPlot=False, plots=None):
        '''
            @params:
                wave: numpy array (n,), containing the waveform to generate the FFT for
                kernel_size: int, between 1 and n, size of the FFT window that will slide over the wave, should be power of 2
                isPlot: bool, if true, produce graphs of output
                plots: list, intervals to be printed if isPlot is true
            @returns:
                list (n//kernel_size, 2) containing, each row contains 2 np arrays of size 
                (kernel_size) containing magnitude and phase angle respectively, i.e. [[magnitude, phaseangle],...]
        '''
        # size = 128 # Size of FFT kernel
        interval = 2  # section of waveform being processed
        if plots:
            plots = set(plots)
        else:
            plots = set()
        wave_length = wave.shape[0]
        last_section = wave_length // kernel_size
        result = np.zeros((last_section, kernel_size))
        for interval in range(last_section):
            section = wave[interval*kernel_size:(interval+1)*kernel_size]
            sp = np.fft.fft(section)
            t = np.arange(sp.shape[0])
            tlen = len(t)//2
            if isPlot and interval in plots:
                plt.clf()
                plt.title("Raw Post-FFT Data, Interval: {}".format(interval))
                plt.plot(t, sp.real, t, sp.imag)
                plt.ylim(-100, 100)
                plt.xlim(-5, tlen)  # Second half of FFT is duplicate info
                plt.show()
                plt.clf()

            # Extract magnitude and angle info
            real = sp.real[:tlen]
            imag = sp.imag[:tlen]
            magnitudes = np.sqrt((real)**2 + (imag)**2)
            angles = np.arctan2(imag, real)
            # print("mag:{}, ang:{}".format(magnitudes.shape, angles.shape))
            row = np.concatenate((magnitudes, angles))
            # print("mag:{}, ang:{}, row:{}".format(magnitudes.shape, angles.shape, row.shape))
            result[interval, :] = row

        return result

    def plot_fft(self, magnitudes, angles, offset=1):
        '''
        Plot the given np arrays of magnitude and phase angle, 
        with an optional offset (i.e. to ignore DC component offset=1)
        '''
        ts = np.arange(magnitudes.shape[0])[offset:]
        # Ignoring DC component which is very large
        plt.plot(ts, magnitudes[offset:])
        plt.title("Magnitudes Squared")
    #     plt.ylim(-100, 100)
        plt.show()
        plt.clf()

        plt.plot(ts, angles[offset:])
        plt.title("Phase Angles")
        plt.show()
        plt.clf()
