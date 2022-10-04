# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:24:09 2022

@author: mbp
"""

import numpy as np
from scipy.signal import hann, lfilter
from scipy.linalg import solve_toeplitz
import network as net
eps = np.finfo(float).eps

def stdft(x, frame_size, overlap, fft_size):
    '''
    compute the short term fourier transform of x using the Hann window
    
    x: input signal
    frame_size: size of the sliding window
    overlap: number of samples to shift the window
    fft_size: size of the fft window
    '''
    frames = np.arange(0, len(x)-frame_size+1, overlap)
    x_stdft = np.zeros((len(frames), fft_size), dtype=np.complex128)
    
    w = hann(frame_size+1, False)[1:] # Omitting the first entry yields the same window as Matlab hanning(frame_size)
    
    for i in range(len(frames)):
        ii = np.arange(frames[i], frames[i] + frame_size)
        x_stdft[i, :] = np.fft.fft(x[ii]*np.sqrt(w), n=fft_size)
    return x_stdft

def thirdoct(fs, fft_size, numBands, mn):
    '''
    compute the 1/3 octave band matrix
    
    fs: sampling frequency
    fft_size: size of the fft window
    numBands: number of desired 1/3 octave bands
    mn: the first band center frequency
    
    A: 1/3 octave band matrix
    cf: center frequency vector
    flr: the left and right edge bands
    '''
    f = np.linspace(0, fs, fft_size+1)
    f = f[:int(fft_size/2+1)]
    k = np.arange(numBands)
    cf = 2**(k/3)*mn
    
    fl = np.sqrt((2**(k/3)*mn) * 2**((k-1)/3)*mn)
    fr = np.sqrt((2**(k/3)*mn) * 2**((k+1)/3)*mn)
    flr = []
    A = np.zeros((numBands, len(f)))
    
    for i in range(len(cf)):
        b = np.argmin((f-fl[i])**2)
        fl[i] = f[b]
        fl_ii = b
        
        b = np.argmin((f-fr[i])**2)
        fr[i] = f[b]
        fr_ii = b
        A[i, np.arange(fl_ii, fr_ii)] = 1
        
        flr.append((fl_ii,fr_ii))
    
    rnk = np.sum(A, axis = 1)
    numBands = np.nonzero(np.logical_and((rnk[1:] >= rnk[:-1]), (rnk[1:] != 0)) != 0)[-1][-1]+2 # This is supposed to remove bands that end up empty because of e.g. low sampling frequency.
    A = A[:numBands, :]
    cf = cf[:numBands]
    return A, cf, flr

def transform_matrix(x, A, frame_size, overlap, fft_size=512):
    '''
    compute the 1/3 octave bands of x
    
    x: input signal
    A: 1/3 octave band matrix
    frame_size: size of the sliding window
    overlap: number of samples to shift the window
    fft_size: size of the fft window
    
    X: The 1/3 octave band representation of x
    '''
    numBands = np.size(A, axis=0)
    x_hat = stdft(x, frame_size, overlap, fft_size)
    x_hat = x_hat[:,:int(fft_size/2 + 1)].T
    
    X = np.zeros((numBands, np.size(x_hat, axis=1)))
    
    for i in range(np.size(x_hat, axis=1)):
        X[:,i] = np.sqrt(np.dot(A, np.abs(x_hat[:,i])**2)) # A is sparse so this may be slower than necessary
    
    return X

def thirdoctave_transform(x, A, flr, frame_size, overlap, fft_size=512, stft=True):
    '''
    compute the 1/3 octave bands of x without the use of matrix-vector products
    
    x: input signal
    A: 1/3 octave band matrix
    flr: left and right indices of octave bands in the stdft
    frame_size: size of the sliding window
    overlap: number of samples to shift the window
    fft_size: size of the fft window
    
    X: The 1/3 octave band representation of x
    '''
    numBands = np.size(A, axis=0)
    if stft:
        x_hat = stdft(x, frame_size, overlap, fft_size)
        x_hat = x_hat[:,:int(fft_size/2 + 1)].T
    else:
        x_hat = x[:,]
    
    X = np.zeros((numBands, np.size(x_hat, axis=1)), dtype=np.complex)
    
    for i in range(numBands):
        if stft:
            X[i,:] = np.sqrt(np.sum(np.abs(x_hat[flr[i][0]:flr[i][1], :])**2, axis=0))
        else:
            band = x_hat[flr[i][0]:flr[i][1], :]
            X[i,:] = np.mean(band, axis=0)
    return X

def sliding_window(x, win_len, stride, axis=0):
    '''
    implements a sliding window of configurable length and stride along the given axis of an array
    '''
    N_win = int((1 + x.shape[axis] - win_len)/stride)
    I = np.zeros((N_win, win_len), dtype=np.int)
    for i in range(win_len):
        I[:,i] = np.arange(i, (N_win)*stride + i, stride)
    return x.take(I, axis)

def pre_process(x):
    '''
    Preprocessing of s time-domain waveform speech signal.
    
    x: noisy/processed speech signal.
    '''
    frame_size = 2**8
    overlap = 2**7
    fft_size = 2**8
    
    X = stdft(x, frame_size, overlap, fft_size)[:,:int(fft_size/2 + 1)]
    X = np.expand_dims(np.expand_dims(X, axis=0), axis=3)
    return X

def load_Network_weights(weights):
    '''
    Load weights for the network with or without 
    
    weights: path to the file containing the trained weights
    dsmf: (bool) if True then load the DSMF's
    '''
    
    arch_args = {'Q':129, 'K':128, 'K_size':(3,3), 'structure':np.array((0, 0, 0, 0, 0, 0, 0, 0)), 'dropout':True}
    model = net.DeepSpeechPresence_ResNet(**arch_args)

    
    model.compile(optimizer='Adam', loss='mse')
    randx = np.random.randn(1, 128, 129, 1)
    randy = np.random.randn(1, 128, 129, 1)
    model.fit(randx, randy, epochs=1, verbose=False)
    model.load_weights(weights)
    return model

def post_process(X, N=30, p=0.05):
    '''
    Top-p percent post-processing of network output.
    
    X: Network output
    N: window length
    p: top-percentage
    '''
    X_full = np.copy(X.T)
    X_slices = []
    for i in range(int(X_full.shape[1]/1000)+1):
        X = sliding_window(X_full[:,i*1000:(i+1)*1000], N, stride=5, axis=1)
        X = np.transpose(X, (1,0,2))
        X = np.reshape(X, (X.shape[0], -1), 'F')
        X = np.sort(X, axis=1)[:, ::-1]
        X_slices.append(X[:, :int((p)*X.shape[1])])
    return np.mean(np.concatenate(X_slices))

# =============================================================================
# noise generation
# =============================================================================

def remove_silent_frames(x, y, r, frame_size, overlap, extra=None):
    frames = np.arange(0, len(x)-frame_size+1, overlap)    
    w = hann(frame_size+1, False)[1:] # Omitting the first entry yields the same window as Matlab hanning(frame_size)
    msk = np.zeros(len(frames))
    
    for j in range(len(frames)):
        jj = np.arange(frames[j], frames[j] + frame_size)
        msk[j] = 20*np.log10(np.linalg.norm(x[jj]*w)/np.sqrt(frame_size))
        
    msk = (msk - np.max(msk) + r) > 0
    count = 0
    
    x_sil = np.zeros(np.size(x))
    y_sil = np.zeros(np.size(y))
    if type(extra) != type(None):
        e_sil = np.zeros(np.size(extra))
    
    for j in range(len(frames)):
        if msk[j]:
            jj_i = np.arange(frames[j], frames[j]+frame_size)
            jj_o = np.arange(frames[count], frames[count]+frame_size)
            x_sil[jj_o] = x_sil[jj_o] + x[jj_i]*w
            y_sil[jj_o] = y_sil[jj_o] + y[jj_i]*w
            if type(extra) != type(None):
                e_sil[jj_o] = extra[jj_i]
            count = count+1
    
    x_sil = x_sil[np.arange(jj_o[-1])]
    y_sil = y_sil[np.arange(jj_o[-1])]
    if type(extra) != type(None):
        e_sil = e_sil[np.arange(jj_o[-1])]
        return x_sil, y_sil, e_sil
    
    return x_sil, y_sil

def checkerboard_mask(dim, checker_size):
    checkerboard = np.zeros(dim)
    for i in range(checker_size):
        for j in range(checker_size):
            checkerboard[i::checker_size*2, j::checker_size*2] = 1.0
            checkerboard[i+checker_size::checker_size*2, j+checker_size::checker_size*2] = 1.0
    return checkerboard

def compute_ssn_filter(speech, order):
    filter_coeff = np.ones(order+1)
    corr = np.correlate(speech, speech, mode='full')
    auto_corr = corr[int(corr.shape[0]/2):int(corr.shape[0]/2)+order+1]
    filter_coeff[1:] = solve_toeplitz(auto_corr[:-1], -auto_corr[1:])
    return filter_coeff

def white_to_ssn(white_noise, a=np.array((1, -4.43922491966070, 8.28069940223227, -6.98329947913563, -1.07949564657074, 9.14188496513308, -9.15568792567125, 1.53993375518200, 5.85546814034682, -7.39180354054988, 4.52327264958545,-1.51728415785012, 0.225723879717352)), b=np.array((1, ))):
    ssn = lfilter(b, a, white_noise)
    return ssn

def gen_ssn(duration, snr, *args):
    raw_noise = np.random.randn(duration)
    raw_noise = raw_noise - np.mean(raw_noise)
    raw_noise = raw_noise / np.sqrt(np.var(raw_noise))
    ssn = white_to_ssn(raw_noise, *args)
    ssn = ssn - np.mean(ssn)
    ssn = ssn / np.sqrt(np.var(ssn))
    scaled_ssn = ssn * np.sqrt(10**(-snr/10))
    return scaled_ssn

def gen_harmonic(duration, f0, decay, modulation):
    signal = np.zeros(duration)
    t = np.linspace(0, duration/10000, duration)
    harmonics = np.arange(f0, 10000, f0)
    amp = 1.0
    for f in harmonics:
        signal += np.cos(t*2*np.pi*f)*amp
        amp *= decay
    return signal*(np.cos(t*np.pi*2*modulation)/2 + 0.5)