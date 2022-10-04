# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:31:27 2022

@author: mbp
"""

import numpy as np
from scipy.signal import resample_poly
from scipy.io.wavfile import read
from functions import pre_process, load_Network_weights, post_process

# input your .wav file path here
x_path = 'my_speech_file.wav'

weights_path = 'Network_weights.h5'


rate, x = read(x_path)
if len(x.shape) > 1:
    x = x[:,0]
x = resample_poly(x, 10000, rate)
X = pre_process(x)

model = load_Network_weights(weights_path)

# compute network output.
X = np.squeeze(model.predict(X))

# apply top-5 percent average post-processing stage
i = post_process(X)

print('Predicted SI: {:.3}'.format(i))