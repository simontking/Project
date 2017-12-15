#!/usr/bin/env python3
from numpy import *
import soundfile as sf

sampleRate = 44100       # sampling rate, Hz, must be integer
duration = 0.02   # in seconds, may be float
frequency = 10000.0        # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
def generateSample(frequencyL, frequencyR, sampleRate, duration, filename):
    sample = empty(((sampleRate*duration),2))
    sample[:,0] = (sin(2*pi*arange(sampleRate*duration)*frequencyL/sampleRate))
    sample[:,1] = (sin(2*pi*arange(sampleRate*duration)*frequencyR/sampleRate))
    sf.write(filename + '.wav', sample, sampleRate)
    return sample
