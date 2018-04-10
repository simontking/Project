#!/usr/bin/env python3
import argparse
import queue
import sys
import wave
import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wf
import soundfile as sf
import sounddevice as sd
import tempfile
import threading
import time
import findlag 
import signalgeneration as sig

audio = pyaudio.PyAudio()
sd.default.device = 'Samson C03U','bcm2835 ALSA: - (hw:2,0)'
devicePi = 'bcm2835 ALSA: - (hw:2,0)'
argsinterval=30
argsblocksize=256
argsbuffersize=256
argssamplerate=44100
argsdownsample=10
argschannels=1

micq = queue.Queue(maxsize=argsbuffersize)
scq = queue.Queue(maxsize=argsbuffersize)
piq = queue.Queue(maxsize=argsbuffersize)

p = pyaudio.PyAudio()
fig, (ax1,ax2) = plt.subplots(1,2)
plt.ion

volume = 1.0     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 0.05   # in seconds, may be float
f = 8000.0        # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
silence = np.zeros(int(fs * 0.1))
paddedSample = np.append(silence, samples)
paddedSample = np.append(paddedSample, silence)
sf.write('sine440.wav', paddedSample, fs)

def mic_audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    if any(indata):
        micq.put(indata.copy())

def sc_audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    if any(indata):
        scq.put(indata.copy())

def pi_audio_callback(outdata, frames, time, status):
    #print("pi_CB")
    """This is called (from a separate thread) for each audio block."""
    assert frames == argsblocksize
    if status.output_underflow:
        print('Output underflow: increase blocksize?', file=sys.stderr)
        raise sd.CallbackAbort
    assert not status
    try:
        data = piq.get_nowait()
    except queue.Empty:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort
    if len(data) < len(outdata):
        outdata[:len(data)] = data
        outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
        raise sd.CallbackStop
    else:
        outdata[:] = data

def measure(vlatency):
    event = threading.Event()
    micdata =  np.zeros(1)
    scdata =  np.zeros(1)
    micstream = sd.InputStream(
        device=deviceMic, channels=argschannels,
        samplerate=argssamplerate, callback=mic_audio_callback)
    scstream = sd.InputStream(
        device=deviceSoundcard, channels=argschannels,
        samplerate=argssamplerate, callback=sc_audio_callback)
    with sf.SoundFile('sine440.wav') as f:
        for _ in range(argsbuffersize):
            data = f.buffer_read(argsblocksize, dtype='float32')
            if not data:
                break
            piq.put_nowait(data)  # Pre-fill queue
            
            stream = sd.RawOutputStream(
                samplerate=f.samplerate, blocksize=argsblocksize,
                device=devicePi, channels=f.channels, dtype='float32',
                callback=pi_audio_callback, finished_callback=event.set)
            with stream:
                timeout = argsblocksize * argsbuffersize / f.samplerate       
                with micstream:
                    with scstream:
                        while data:
                            data = f.buffer_read(argsblocksize, dtype='float32')
                            piq.put(data, timeout=timeout)
                            micdata = np.append(micdata, micq.get())
                            scdata = np.append(scdata, scq.get())
                        while not event.is_set():
                            micdata = np.append(micdata, micq.get())
                            scdata = np.append(scdata, scq.get())
    with open('micdata', 'wb') as micf:
        np.save(micf, micdata)
    with open('scdata', 'wb') as scf:
        np.save(scf, scdata)
    timediff, distance = findlag.measure(scdata, micdata, 44100, vlatency)
    with piq.mutex:
        piq.queue.clear()
    with scq.mutex:
        scq.queue.clear()
    with micq.mutex:
        micq.queue.clear()
    return timediff, distance

def latency():
    event = threading.Event()
    micdata =  np.zeros(1)
    scdata =  np.zeros(1)
    micstream = sd.InputStream(
        device=deviceMic, channels=argschannels,
        samplerate=argssamplerate, callback=mic_audio_callback)
    scstream = sd.InputStream(
        device=deviceSoundcard, channels=argschannels,
        samplerate=argssamplerate, callback=sc_audio_callback)
    with sf.SoundFile('sine440.wav') as f:          
        for _ in range(argsbuffersize):
            data = f.buffer_read(argsblocksize, dtype='float32')
            if not data:
                break
            piq.put_nowait(data)  # Pre-fill queue
            
            stream = sd.RawOutputStream(
                samplerate=f.samplerate, blocksize=argsblocksize,
                device=devicePi, channels=f.channels, dtype='float32',
                callback=pi_audio_callback, finished_callback=event.set)
            with stream:
                timeout = argsblocksize * argsbuffersize / f.samplerate       
                with micstream:
                    with scstream:
                        while data:
                            data = f.buffer_read(argsblocksize, dtype='float32')
                            piq.put(data, timeout=timeout)
                            micdata = np.append(micdata, micq.get())
                            scdata = np.append(scdata, scq.get())
                        while not event.is_set():
                            micdata = np.append(micdata, micq.get())
                            scdata = np.append(scdata, scq.get())
    timediff = findlag.correlate(scdata, micdata, 44100)
    with piq.mutex:
        piq.queue.clear()
    with scq.mutex:
        scq.queue.clear()
    with micq.mutex:
        micq.queue.clear()
    return timediff


def simple_transceiver():
    print('yo')
    sample = sig.generateSample(10000, 10000, 44100, 0.1, 'sample')
    samplePad = np.append(sample,np.zeros(44100))
    micdata  = sd.playrec(samplePad, samplerate=44100, channels=2, dtype='float32')
    sd.wait()
    print(micdata.shape)
    d=micdata.sum(axis=1)/2
    timediff,acor,distance = findlag.measureTimeOfArrival(sample.sum(axis=1)/2, d, 44100)
    with open('micdata', 'wb') as micf:
        np.save(micf, micdata)
    with open('sampledata', 'wb') as sf:
        np.save(sf, sample)
    
    ax1.clear()
    ax1.plot(np.arange(0,len(micdata)),micdata)
    ax2.clear()
    ax2.plot(np.arange(0,len(sample)),sample)
    plt.show()
    return timediff

def stream_callback(indata, outdata, frames, time, status):
    #print("pi_CB")
    """This is called (from a separate thread) for each audio block."""
    assert frames == argsblocksize
    if status.output_underflow:
        print('Output underflow: increase blocksize?', file=sys.stderr)
        raise sd.CallbackAbort
    assert not status
    try:
        data = piq.get_nowait()
    except piq.Empty:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort
    if len(data) < len(outdata):
        outdata[:len(data)] = data
        outdata[len(data):] = np.zeros(((len(outdata) - len(data)),2))
        raise sd.CallbackStop
    else:
        outdata[:] = data
    if len(indata)>0:
        micq.put(indata.copy())

def stream_transceiver():
    event = threading.Event()
    sample = sig.generateSample(10000, 10000, 44100, 0.1, 'sample')
    micdata =  np.zeros(1)
    scdata =  np.zeros(1)
    with piq.mutex:
        piq.queue.clear()
    with micq.mutex:
        micq.queue.clear()
    micstream = sd.Stream(channels=(2,2), samplerate=argssamplerate, callback=stream_callback,
                          blocksize=argsblocksize, device = ['Samson C03U','bcm2835 ALSA: - (hw:2,0)'],
                          finished_callback=event.set)
    with sf.SoundFile('sample.wav','r') as f:
        while f.tell() < len(f):
            data = f.read(argsblocksize, dtype='float32')
            if len(data) <= 0:
                break
            piq.put_nowait(data)
            with micstream:
                timeout = argsblocksize * argsbuffersize / f.samplerate   
                while f.tell() < len(f):
                    data = f.read(argsblocksize, dtype='float32')
                    piq.put(data, timeout=timeout)
                    micdata = np.append(micdata, micq.get())
                while not event.is_set():
                    micdata = np.append(micdata, micq.get())
                event.wait()
    with open('micdata', 'wb') as micf:
        np.save(micf, micdata)
    timediff, distance = findlag.measure(sample, micdata, 44100, vlatency)
    with piq.mutex:
        piq.queue.clear()
    with micq.mutex:
        micq.queue.clear()
    return timediff, distance

def get_latency():
    try:
        lagArray = []
        for i in range(10):
            x, i = latency()
            lagArray.append(x)
        xLat = np.mean(lagArray)
        print (xLat)
        return xLat
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except queue.Full:
        # A timeout occured, i.e. there was an error in the callback
        print("Queue Full")
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))
        
def output():
    event = threading.Event()
    with sf.SoundFile('sine440.wav') as f:          
        for _ in range(argsbuffersize):
            data = f.buffer_read(argsblocksize, dtype='float32')
            if not data:
                break
            piq.put_nowait(data)  # Pre-fill queue
            
            stream = sd.RawOutputStream(
                samplerate=f.samplerate, blocksize=argsblocksize,
                device=devicePi, channels=f.channels, dtype='float32',
                callback=pi_audio_callback, finished_callback=event.set)
            with stream:
                timeout = argsblocksize * argsbuffersize / f.samplerate       
                while data:
                    data = f.buffer_read(argsblocksize, dtype='float32')
                    piq.put(data, timeout=timeout)
                event.wait()

x = simple_transceiver()
print('ok')