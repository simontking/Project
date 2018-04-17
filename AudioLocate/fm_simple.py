import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, remez, lfilter, decimate

fbw = 4.0
carrier_frequency = 40.0
modulation_index = 1.0
fs=44100

time = np.arange(44100.0) / 44100.0
modulator = np.sin(2.0 * np.pi * fbw * time) * modulation_index
carrier = np.sin(2.0 * np.pi * carrier_frequency * time)
product = np.zeros_like(modulator)

for i, t in enumerate(time):
    product[i] = np.sin(2. * np.pi * (carrier_frequency * t + modulator[i]))

fc1 = np.exp(-1j*2*np.pi*(carrier_frequency)*time)
print(fc1)
x2 = np.array(product * fc1).astype("complex64")
lpfr = remez(64, [0,fbw,fbw+(fs/2 - fbw)/4, fs/2], [1,0], Hz=fs)
x3 = lfilter(lpfr,1.0,x2)

dec_rate = 5
print('dec',dec_rate)
x4 = x3[0::dec_rate]
fs_y = fs/dec_rate
print('fsy',fs_y)
x42 = decimate(x2, dec_rate)
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(product)
plt.subplot(4, 1, 2)
plt.plot(x2)
plt.subplot(4, 1, 3)
plt.plot(x3)
plt.subplot(4, 1,4)
plt.plot(x4)

tmp = np.array(x4[1:]*np.conj(x4[:-1])).astype("complex64")
x5=np.angle(tmp)

d = fs_y * 75e-6
x = np.exp(-1/d)
b=[1-x]
a=[1,-x]
x6 = lfilter(b,a,x5)

dec_audio = int(fs_y/fs)
print(dec_audio)

x7 = decimate(x6, 5)
print(x7)
x8 = x7*10000/np.max(np.abs(x7))
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(tmp)
plt.subplot(4, 1, 2)
plt.plot(x6)
plt.subplot(4, 1, 3)
plt.plot(x7)
plt.subplot(4, 1,4)
plt.plot(x8)
plt.show()