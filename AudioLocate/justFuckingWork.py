import numpy as np
import matplotlib.pyplot as plt
import signalgeneration as sig
from scipy.signal import hilbert, remez, lfilter, decimate

fc = 150.0
modulation_index = 1.0
fs = 1500.0
sample_period = 1/fs
ZC_N = 15
fbw = 25
zcsig = sig.generateZCSample(ZC_N ,1)
print (zcsig)

time = np.arange(fs) / fs

carrier = np.cos(2.0 * np.pi * fc * time)
real_modulator = np.zeros_like(carrier).astype("complex64")
real_product = np.zeros_like(carrier)
imag_modulator = np.zeros_like(carrier).astype("complex64")
imag_product = np.zeros_like(carrier)
product = np.zeros_like(carrier)
# Uspsample
for i, t in enumerate(time):
    x = np.int(i/(fs/ZC_N))
    real_modulator[i] = zcsig.real[x]
    imag_modulator[i] = zcsig.imag[x]
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(zcsig)
plt.subplot(3, 1, 2)
plt.plot(real_modulator,'r')
plt.plot(imag_modulator,'b')
for i, t in enumerate(time):
    real_product[i] = np.cos((2.0 * np.pi * (fc+(imag_modulator[i]*fbw)) * t))* real_modulator[i]
    #imag_product[i] = np.sin((2.0 * np.pi * fc * t))* imag_modulator[i]
plt.subplot(3, 1, 3)
plt.plot(carrier,'b')
plt.plot(real_product,'r')

product = (real_product - imag_product)

##for i, t in enumerate(time):
##    x = np.int(i/(fs/ZC_N))
##    product[i] = zcsig[x]  * np.exp(-1j*2*np.pi*(fc)*t)
##    
product_demod = np.zeros_like(product).astype("complex64")
chap = np.zeros_like(product).astype("complex64")

real_result = (np.cos((2.0 * np.pi * (fc) * t)) * product) 
imag_result = -(np.sin((2.0 * np.pi * (fc) * t)) * product) 
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(product)
plt.subplot(3, 1, 2)
plt.plot(real_product,'b')
plt.plot(real_result,'r')
plt.subplot(3, 1,3)
plt.plot(imag_product)
plt.plot(imag_result)


tmp = product[1::1] * np.conjugate(product[0:-1:1]);
# Record the angle of the complex difference vectors

##plt.figure()
##plt.subplot(2, 1, 1)
##plt.plot(tmp)
##plt.subplot(2, 1, 2)
##plt.plot(np.unwrap(np.angle(tmp)))

z = hilbert(product)
zr= z
inst_ampl = np.abs(zr)
inst_phase = np.unwrap(np.angle(zr))
inst_freq = np.diff(inst_phase)/(2*np.pi)*fs
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(inst_ampl)
plt.subplot(4, 1, 2)
plt.plot(inst_phase)
plt.subplot(4, 1,3)
plt.plot(inst_freq - fc)
plt.subplot(4, 1, 4)
plt.plot(np.cos(inst_phase))

rzc = np.zeros(ZC_N)
for k in range(ZC_N):
    demod_block = inst_ampl[k*np.int(len(inst_ampl)/ZC_N):(1+k)*np.int(len(inst_ampl)/ZC_N)]
    rzc[k] = np.average(demod_block)

plt.figure()
plt.plot(zcsig,'b')
plt.plot(rzc,'r')
##z = hilbert(imag_result)
##inst_ampl = np.abs(z)
##inst_phase = np.unwrap(np.angle(z))
##inst_freq = np.diff(inst_phase)/(2*np.pi)*fs
##plt.figure()
##plt.subplot(4, 1, 1)
##plt.plot(inst_ampl)
##plt.subplot(4, 1, 2)
##plt.plot(inst_phase)
##plt.subplot(4, 1,3)
##plt.plot(inst_freq)
##plt.subplot(4, 1, 4)
##plt.plot(np.cos(inst_phase))

plt.show()
