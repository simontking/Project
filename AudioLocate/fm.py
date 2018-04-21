import numpy as np
import matplotlib.pyplot as plt
import signalgeneration as sig
from scipy.signal import hilbert, remez, lfilter, decimate

fc = 150.0
modulation_index = 1.0
fs = 1500.0
sample_period = 1/fs
ZC_N = 7
fbw = 15

zcsig = sig.generateZCSample(ZC_N ,1)
print (zcsig)

time = np.arange(fs) / fs

carrier = np.cos(2.0 * np.pi * fc * time)
real_modulator = np.zeros_like(carrier)
real_product = np.zeros_like(carrier)
product = np.zeros_like(carrier)

imag_modulator = np.zeros_like(carrier)
imag_product = np.zeros_like(carrier)

for i, t in enumerate(time):
    x = np.int(i/(fs/ZC_N))
    real_modulator[i] = zcsig.real[x]
    imag_modulator[i] = zcsig.imag[x]

for i, t in enumerate(time):
    real_product[i] = np.cos((2.0 * np.pi * fc * t) + (real_modulator[i]*(fbw)))
    imag_product[i] = np.sin((2.0 * np.pi * fc * t) + (imag_modulator[i]*(fbw))) 

product = (real_product + imag_product)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(zcsig)
plt.subplot(3, 1, 2)
plt.plot(real_product,'r')
plt.plot(imag_product,'b')
plt.subplot(3, 1,3)
plt.plot(product)

##for i, t in enumerate(time):
##    x = np.int(i/(fs/ZC_N))
##    product[i] = zcsig[x]  * np.exp(-1j*2*np.pi*(fc)*t)
##    
product_demod = np.zeros_like(product).astype("complex64")
chap = np.zeros_like(product).astype("complex64")

real_result = np.cos((2.0 * np.pi * (fc) * t)) * product
z = hilbert(real_result)
inst_ampl = np.abs(z)
inst_phase = np.unwrap(np.angle(z))
inst_freq = np.diff(inst_phase)/(2*np.pi)*fs
##plt.figure()
##for v in zcsig:
##    plt.polar([0,np.angle(v)], [0,np.abs(v)], marker='o')
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(inst_ampl)
plt.subplot(3, 1, 2)
plt.plot(inst_phase)
plt.subplot(3, 1,3)
plt.plot(inst_freq)

imag_result = np.sin((2.0 * np.pi * (fc) * t)) * product
z = hilbert(imag_result)
inst_ampl = np.abs(z)
inst_phase = np.unwrap(np.angle(z))
inst_freq = np.diff(inst_phase)/(2*np.pi)*fs
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(inst_ampl)
plt.subplot(3, 1, 2)
plt.plot(inst_phase)
plt.subplot(3, 1,3)
plt.plot(inst_freq)
plt.show()

#real_result = np.cos((2.0 * np.pi * (fc) * t)) * product
#imag_result = np.sin((2.0 * np.pi * (fc) * t)) * 
for i, t in enumerate(time):
    #x = np.int(i/(fs/ZC_N))
    product_demod[i] = product[i]  * np.exp(-1j*2*np.pi*(fc)*t)
plt.figure()
result = np.zeros(ZC_N).astype("complex64")
for k in range(ZC_N):
    demod_block = product_demod[k*np.int(len(product_demod)/ZC_N):(1+k)*np.int(len(product_demod)/ZC_N)]
    plt.subplot(ZC_N, 2, 2*k+1)
    plt.plot(demod_block)
    fftinfo = np.abs(np.fft.fft(demod_block))
    chap[k*np.int(len(product_demod)/ZC_N):(1+k)*np.int(len(product_demod)/ZC_N)] = fftinfo
    plt.subplot(ZC_N, 2, 2*k+2)
    plt.plot(fftinfo)
    I = np.argmax(fftinfo)
    tmp = demod_block[1::1] * np.conjugate(demod_block[0:-1:1]);
  # Record the angle of the complex difference vectors
    print(  np.angle(tmp));
    result[k] = (np.abs(fftinfo[I])/(fs/fc)) + 1j*np.angle(fftinfo[I])
    
print(result)
fc1 = np.exp(-1j*2*np.pi*(fc)*time)
#print(fc1)
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
plt.plot(product_demod)
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
#print(x7)
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