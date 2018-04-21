import numpy 
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.special import erfc
import sys
import signalgeneration as sig

#size of constellation (N symbols per frame; N frames per constellation)
N=7 #number of data symbols
k=1 #
A=1
l=4
#x=2*numpy.random.random_integers(0,1,(N,N))-1 #real part of symbol matrix 's'
#y=2*numpy.random.random_integers(0,1,(N,N))-1 #imaginary part of symbol matrix 's'

s=x+1j*y #complex matrix (x+y) of QPSK symbols

s = sig.generateZCSample(N,1)
print('s=%s',s)
y= numpy.empty((N,l), dtype=complex) #generate two empty, NxN arrays for use later

# definitions for the plot axe
left, width=0.1,0.65
bottom, height=0.1,0.65
bottom_h=left_h=left+width+0.02

rect_scatter=[left,bottom,width,height]
rect_histx=[left,bottom_h,width,0.2]
rect_histy=[left_h,bottom,0.2,height]

# start with a rectangular figure
plt.figure(1, figsize=(8,8))

# set up plots
axScatter=plt.axes(rect_scatter)
axHistx=plt.axes(rect_histx)
axHisty=plt.axes(rect_histy)

# no axis labels for box plots
axHistx.xaxis.set_major_formatter(NullFormatter())
axHisty.yaxis.set_major_formatter(NullFormatter())

# Generate SNR scale factor for AWGN generation:
error_sum=0.0 # initialize counter to zero to be used in BER calculation

SNR_MIN=-10
SNR_MAX=10
SNR=SNR_MAX # desired SNR used to determine noise power

Eb_No_lin=10**(SNR/10.0) # convert SNR to decimal

print('Eb_No_lin=%s',Eb_No_lin)

No=1.0/Eb_No_lin # Linear power of the noise; average signal power = 1 (0dB)
scale=numpy.sqrt(No/2) # variable to scale random noise values in AWGN loop

print('No=%s',No)
print('scale=%s',scale)

# loop through each frame, modulate, add gaussian noise (AWGN) then decode back in symbols
for i in range(N):
	#n=numpy.fft.ifftn(numpy.random.normal(scale=scale, size=N)+1j*numpy.random.normal(scale=scale, size=N)) # array of noise
	n=0 # uncomment here and comment above if you want to remove all noise
	#print('n[%s]=\n%s',i,n)
	t[i]=numpy.fft.ifftn(s[i])
	print(t[i])
	w[i]=numpy.fft.fftn(t[i]+n) # add noise here
	
	# decode received signal + noise back into bins/symbols
	z=numpy.sign(numpy.real(w[i]))+1j*numpy.sign(numpy.imag(w[i]))
	#print('z of loop %s=\n%s',i,z)
	#print('z!=s[%s]=\n%s',i,z!=s[i])

	# find errors
	err=numpy.where(z != s[i])
	#print('err[%s]=\n%s',i,err)
	
	# add up errors per frame
	error_sum+=float(len(err[0]))
	#print('error_sum[%s]=\n%s',i,error_sum)

# show total error for entire NxN message
BER=error_sum/N**2
print('Final error_sum = %s out of a total possible %s symbols',error_sum,N**2)
print('Total BER=%s',BER)

# scatter plot:
axScatter.scatter(numpy.real(w),numpy.imag(w))

# draw axes at origin
axScatter.axhline(0, color='black')
axScatter.axvline(0, color='black')

# add title (at x-axis) to scatter plot
#title = 'Zero noise'
title = 'SNR = %sdB with a BER of %s' % (SNR,BER)
axScatter.xaxis.set_label_text(title)

# now determine nice limits by hand:
binwidth = 0.25 # width of histrogram 'bins'
xymax = numpy.max( [numpy.max(numpy.fabs(numpy.real(w))), numpy.max(numpy.fabs(numpy.imag(w)))] ) # find abs max symbol value; nominally 1 
lim = ( int(xymax/binwidth) + 1) * binwidth # create limit that is one 'binwidth' greater than 'xymax'

axScatter.set_xlim( (-lim, lim) ) # set the data limits for the xaxis -- autoscale
axScatter.set_ylim( (-lim, lim) ) # set the data limits for the yaxis -- autoscale

bins = numpy.arange(-lim, lim + binwidth, binwidth) # create bins 'binwidth' apart between -lin and +lim -- autoscale
axHistx.hist(numpy.real(w), bins=bins) # plot a histogram - xaxis are real values
axHisty.hist(numpy.imag(w), bins=bins, orientation='horizontal') # plot a histogram - yaxis are imaginary values

axHistx.set_xlim( axScatter.get_xlim() ) # set histogram axes to match scatter plot axes limits
axHisty.set_ylim( axScatter.get_ylim() ) # set histogram axes to match scatter plot axes limits

plt.show()