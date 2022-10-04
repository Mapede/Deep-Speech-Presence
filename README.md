# Deep-Speech-Presence
This contains the code needed to run the data-driven speech intelligibility predictor proposed in "Data-Driven Non-Intrusive Speech Intelligibility Prediction using Speech Presence Probability".

network.py contains the Tensorflow 2 architecture. functions.py contains functionality for preprocessing, postprocessing and loading network weights. example.py can be used to run the predictor on a pair of single channel .wav files.

noise_generators.py contains functions for generating the noise types used to train the SPP estimator in the paper. Running some of these generators requires setting a path to a folder containing a few .wav files of speech. These files will be used to generate an SSN filter.
