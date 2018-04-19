#!/usr/bin/env python3
from numpy import *
import soundfile as sf
import sounddevice

sampleRate = 44100  # sampling rate, Hz, must be integer
duration = 0.02  # in seconds, may be float
frequency = 10000.0  # sine frequency, Hz, may be float


# generate samples, note conversion to float32 array
def generateSinSample(frequency, sampleRate, duration, filename):
    sample = empty((int(sampleRate*duration),1))
    sample[:,0] = (sin(2*pi*arange(sampleRate*duration)*frequency/sampleRate))
    sf.write(filename + '.wav', sample, sampleRate)
    return sample

def generateCosSample(frequency, sampleRate, duration, filename):
    sample = empty((int(sampleRate*duration),1))
    sample[:,0] = (cos(2*pi*arange(sampleRate*duration)*frequency/sampleRate))
    sf.write(filename + '.wav', sample, sampleRate)
    return sample

def generateSample(frequencyL, frequencyR, sampleRate, duration, filename):
<<<<<<< HEAD
    sample = empty((int(sampleRate * duration), 2))
    sample[:, 0] = (sin(2 * pi * arange(sampleRate * duration) * frequencyL / sampleRate))
    sample[:, 1] = (sin(2 * pi * arange(sampleRate * duration) * frequencyR / sampleRate))
    sf.write(filename + '.wav', sample, sampleRate)
    return sample


# x = generateSample(3000, 2000, 44100, 2, 'testfile')
# sounddevice.play(x, 44100)
# sounddevice.wait()
# print('done')
=======
    sample = empty((int(sampleRate*duration),2))
    sample[:,0] = (sin(2*pi*arange(sampleRate*duration)*frequencyL/sampleRate))
    sample[:,1] = (sin(2*pi*arange(sampleRate*duration)*frequencyR/sampleRate))
    sf.write(filename + '.wav', sample, sampleRate)
    return sample

def generateZCSample(N,M):
    zcseq = exp((-1j * pi * M * arange(0,N) * arange(1,N+1))/N)
    return zcseq


s= generateSample(5000,5000,44100,0.1,'jerry')

>>>>>>> 950b8d5c925265f51a0d2dd35d00e3884318bd10
