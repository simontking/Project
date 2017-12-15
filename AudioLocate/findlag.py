import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wf
import soundfile as sf
import sounddevice as sd
import tempfile
import threading

def measure(sample, micData, fs, latency):
    timeDiff, acor = correlate(sample, micData, fs)
    speedOfSound = 343
    realDiff = timeDiff - latency
    distance = speedOfSound*realDiff  
    return timeDiff, distance

def measureTimeOfArrival(sample, micData, fs):
    timeDiff, acor = correlate(sample, micData, fs)
    speedOfSound = 343
    distance = speedOfSound*timeDiff  
    print("distance (m): ",distance)  
    return timeDiff, acor, distance

def measureTimeDifferenceOfArrival(sample, firstSample, secondSample, fs):
    f, ((ax1, ax2, ax3)) = plt.subplots(3,1)
    ax1.set_title('Correlation')
    ax2.set_title('Recorded Data')
    ax3.set_title('mic, aligned')
    timeDiff1, acor1 = correlate(sample, firstSample, fs)
    timeDiff2, acor2 = correlate(sample, secondSample, fs)
    timeDiff = timeDiff2 - timeDiff1
    speedOfSound = 343
    distance = speedOfSound*timeDiff
    
    return timeDiff, distance

def correlate(baseSample, data, fs):
    acor = np.correlate(data,baseSample,"full");
    with open('corrdata', 'wb') as micf:
        np.save(micf, acor)
    I = np.argmax(acor)
    timeDiff = (I-(len(baseSample)-1))/fs
    print("max Sample #: ",I,"Time Difference (s): ", timeDiff)
    return timeDiff, acor