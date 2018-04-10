#!/usr/bin/env python3
import numpy as np
import soundfile as sf
import sounddevice as sd
import findlag 
import signalgeneration as sig
import matplotlib.pyplot as plt

sd.default.device = 'Samson C03U','bcm2835 ALSA: - (hw:0,0)'
devicePi = 'bcm2835 ALSA: - (hw:2,0)'
argsinterval=30
argsblocksize=256
argsbuffersize=256
argssamplerate=44100
argsdownsample=10
argschannels=1
sample = sig.generateSample(10000, 10000, 44100, 0.1, 'sample')
samplePad = np.append(sample,np.zeros(44100))
fig, (ax1,ax2) = plt.subplots(2,1)
plt.ion

def simple_transceiver():
    print('yo')
    micdata  = sd.playrec(samplePad, samplerate=44100, channels=2, dtype='float32')
    sd.wait()
    d=micdata.sum(axis=1)/2
    timediff,acor,distance = findlag.measureTimeOfArrival(sample.sum(axis=1)/2, d, 44100)
    with open('initialtest.csv','a') as file:
        x = '{0},{1}\n'.format(timediff,distance )
        file.write(str(x))
    
    return timediff

#for num in range(20):
    #x = simple_transceiver()
#print('ok')
zcsig = sig.generateZCSample(53,1)
t = np.zeros(len(zcsig),dtype=np.complex_)
print (t)
cossig = sig.generateCosSample(10000, 100, 0.01, 'cos')
sinsig = sig.generateSinSample(10000, 100, 0.01, 'sin')
for i in range(len(zcsig)):
	t[i]=np.fft.ifftn(zcsig[i],)
print(t)
sample =  zcsig.real*cossig - zcsig.imag-sinsig
samplePad = np.append(sample,np.zeros(44100))
x = simple_transceiver()
ax1.clear()
ax1.plot(np.arange(0,len(zcsig)),zcsig)
ax2.clear()
ax2.plot(np.arange(0,len(cossig)),cossig)
plt.show()
