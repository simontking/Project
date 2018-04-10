import numpy as np
import matplotlib.pyplot as plt
import signalgeneration as sig

modulator_frequency = 4.0
carrier_frequency = 20.0
modulation_index = 1.0
sample_rate = 10000.0
sample_period = 1/sample_rate
ZC_N = 5
bandwidth = 7.0

zcsig = sig.generateZCSample(ZC_N ,1)
print (zcsig)

time = np.arange(sample_rate) / sample_rate

carrier = np.cos(2.0 * np.pi * carrier_frequency * time)
real_modulator = np.zeros_like(carrier)
real_product = np.zeros_like(carrier)

imag_modulator = np.zeros_like(carrier)
imag_product = np.zeros_like(carrier)

for i, t in enumerate(time):
    x = np.int(i/(sample_rate/ZC_N))
    real_modulator[i] = zcsig.real[x]
    imag_modulator[i] = zcsig.imag[x]

for i, t in enumerate(time):
    real_product[i] = np.cos((2.0 * np.pi * (carrier_frequency + (real_modulator[i]*(bandwidth/2))) * t) )
    imag_product[i] = np.sin((2.0 * np.pi * (carrier_frequency + (imag_modulator[i]*(bandwidth/2))) * t) )

product = (real_product - imag_product) +2

real_result = np.zeros_like(carrier)
imag_result = np.zeros_like(carrier)
T = 1 / carrier_frequency

mod_fact = 1.0 / 100.0
original_theta = 2.0 * np.pi * (carrier_frequency + mod_fact * product) * time
yc = np.cos(original_theta)

theta = np.arccos(yc)

# Calculate the samples per quadrant
N = T / sample_period 
Nq = int(N / 4)

# Solve the angles for Yc's 1st cycle
quadrant1_theta = theta[0:Nq]
quadrant2_theta = np.pi - theta[Nq:Nq*2]
quadrant3_theta = np.pi - theta[Nq*2:Nq*3]
quadrant4_theta = 2.0 * np.pi + theta[Nq*3:int(N)]
theta_corrected = np.hstack((quadrant1_theta, quadrant2_theta, quadrant3_theta, quadrant4_theta))

number_of_cycles = carrier_frequency  

# This is our tally
cycle = 1
pi_multiple = 2.0
cycle_start_index = Nq*4
N=int(N)
while cycle < number_of_cycles:
    
    # Calculate which sample falls into what quadrant
    quad1_start = cycle_start_index
    quad1_end = quad1_start + Nq
    
    quad23_start = quad1_end
    quad23_end = quad23_start + Nq * 2
    
    quad4_start = quad23_end
    quad4_stop = cycle_start_index + N
    
    # Solve for the angles for Yc's 2nd, 3rd, ... cycles
    quadrant1_theta = theta[quad1_start:quad1_end] + pi_multiple * np.pi 
    quadrant23_theta = np.pi - theta[quad23_start:quad23_end] + pi_multiple * np.pi
    quadrant4_theta = 2.0 * np.pi + theta[quad4_start:quad4_stop] + pi_multiple * np.pi 
    theta_corrected = np.hstack((theta_corrected, quadrant1_theta, quadrant23_theta, quadrant4_theta))
    
    cycle = cycle + 1
    
    
    # Calculate a distinct offset for each cycle 
    pi_multiple = pi_multiple + 2.0
    
    cycle_start_index = cycle_start_index + N
    
product_demod = theta_corrected / (2.0 * np.pi * time *  mod_fact) - carrier_frequency / mod_fact

# Remove offset that was added before modulation
product_demod = product_demod - 2

plt.subplot(3, 2, 1)
plt.title('Frequency Modulation')
plt.plot(real_modulator)
plt.ylabel('Amplitude')
plt.xlabel('Modulator signal')
plt.subplot(3, 2, 3)
plt.plot(real_product)
plt.ylabel('Amplitude')
plt.xlabel('Carrier signal')
plt.subplot(3, 2, 2)
plt.plot(imag_modulator)
plt.ylabel('Amplitude')
plt.xlabel('Output signal')
plt.subplot(3, 2, 4)
plt.plot(imag_product)
plt.ylabel('Amplitude')
plt.xlabel('Output signal')
plt.subplot(3, 2, 5)
plt.plot(product)
plt.ylabel('Amplitude')
plt.xlabel('Output signal')
plt.subplot(3, 2, 6)
plt.plot(product_demod)
plt.ylabel('Amplitude')
plt.xlabel('Output signal')
plt.show()