
################################################################################
# Electroglottographic Wavegram Demo
# (C) 2016 Christian T. Herbst - www.christian-herbst.org
################################################################################
#
# How to cite the wavegram software:
#
# Christian T. Herbst, W. T. S. Fitch, Jan G. Svec (2010). Electroglottographic 
# wavegrams: a technique for visualizing vocal fold dynamics noninvasively. 
# J. Acoust. Soc. Am., 128 (5), 3070-3078
#
# Christian T. Herbst (2012). Investigation of glottal configurations in singing. 
# Palacky University Olomouc, the Czech Republic (Doctoral Dissertation)
#
################################################################################
# open a two-channel file (2nd channel is EGG signal) and display a dEGG
# wavegram and a few other pieces of information (including EGG contact 
# quotients) 
################################################################################



################################################################################
# required modules
# Note that the wavegram module requires a couple of other modules which are 
# distributed as part of "Christian's Python Library", available online (and
# documented) at http://www.christian-herbst.org/python/
# You also need the matplotlib (www.matplotlib.org) and numpy (www.numpy.org) 
# distributions
################################################################################
import wavegram
import matplotlib.pyplot as plt
import numpy


################################################################################
# constants/parameters
################################################################################
# stereo WAVE file (left channel: acoustic signal; right channel: EGG signal)
fileName = "D:\\projects\\voice_pathology_ai\\voice_data\\fusion_egg\\healthy\\a\\1-a_n.wav"

# change these values according to the analyzed EGG signal, in order to avoid
# clipping (in which case the colours will be either pure red or pure blue...),
# or use zero for both parameters to enable automatic scaling
forceCycleDetection = True # if False (the default), we'll revert to previously
	# calculated cycle information (if available). set to True if you want to
	# change f0 or cycle detection parameters
graphWidthInches = 10
graphHeightInches = 10
graphResolutionDpi = 72


################################################################################
# start processing
################################################################################

print("WAVEGRAM for file", fileName)
data = wavegram.signal()

# this is the main function, where a lot is happening: fundamental frequeny
# detection and cycle extraction. Please refer to the wavegram module source
# code documentation for the relevance of all parameters (in this demo only
# default parameters are used, but the process can be fine-tuned if needed.
data.wrapWaveFile(fileName)

tStart = 0
tEnd = data.getDuration()

# create the matplotlib figure
plt.clf()
fig = plt.figure(figsize=(graphWidthInches,graphHeightInches), 
	dpi = graphResolutionDpi)

# 1. plot the EGG signal
eggData = data.getRawData(0)
dataT = numpy.zeros(len(eggData))
fs = data.getSamplingFrequency()
for i in range(len(eggData)):
	dataT[i] = i / float(fs)
ax1 = plt.subplot(411)
ax1.plot(dataT, eggData)
ax1.grid(True)
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('EGG waveform')
ax1.set_title(fileName)

# 2. create and plot the wavegram
# The createWavegram method used below has also a few parameters that can be
# changes (defaults used here). Again, please refer to the wavegram module 
# source code documentation for the meaning of these.
waveGramData1 = data.createWavegram(derivative = True)
ax2 = plt.subplot(412, sharex=ax1)
ax2.imshow(waveGramData1, interpolation='nearest', aspect='auto', 
	origin='lower', extent=[tStart, tEnd, 0, 100])
ax2.grid(True)
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('DEGG wavegram')

# 3. plot the fundamental frequency contour
arrF0 = data.metaData.arrF0
ax3 = plt.subplot(413, sharex=ax1)
ax3.plot(arrF0['t'], arrF0['freq'], linewidth = 3)
ax3.set_ylabel('F0 [Hz]')
ax3.set_xlabel('Time [sec]')
ax3.grid(True)
if tEnd == -1:
	tEnd = arrF0['t'][-1]
ax3.set_xlim(tStart, tEnd)

# 4. plot the EGG contact quotient, using five different algorithms: three
# threshold-based algorithms; a dEGG algorithm (looking for the positive and
# the negative peak in the 1st derivative of the EGG signal [dEGG]); and a hybrid 
# approach where the contacting event is derived from the positive dEGG peak, 
# and the de-contacting event is determined using a threshold (typically set at
# 3/7)
ax4 = plt.subplot(414, sharex=ax1)
arrCqMetaData = [
	['threshold 20%', wavegram.CQ_THRESHOLD, 0.2],
	['threshold 25%', wavegram.CQ_THRESHOLD, 0.25],
	['threshold 35%', wavegram.CQ_THRESHOLD, 0.35],
	['dEGG', wavegram.CQ_DEGG, None],
	['dEGG', wavegram.CQ_HYBRID, 3.0 / 7.0], 
]
arrDataCQ = []
arrT = None
for algoIdx, cqMetaData in enumerate(arrCqMetaData):
	label = cqMetaData[0]
	method = cqMetaData[1]
	threshold = cqMetaData[2]
	arrT, arrCQ, arrContactingOffset, arrDecontactingOffset = \
		data.calculateCQ(method = method, threshold = threshold, 
			doInterpolate = True)
	ax4.plot(arrT, arrCQ, label=label)
	arrDataCQ.append(arrCQ)

ax4.legend(loc='best', fancybox=True, framealpha=0.7, fontsize=9)
ax4.set_xlim(tStart, tEnd)
ax4.grid()
ax4.set_ylim(-0.05, 1.05)
ax4.set_xlabel('Time [sec]')
ax4.set_ylabel('$CQ_{EGG}$')



# finalize and save the graph
plt.tight_layout()
fileNameOnly = '.'.join(fileName.split('.')[:-1])
plt.savefig(fileNameOnly + '_wavegram.png')


# create a CVS file with the EGG_CQ data:
csvFileName = "CQ_EGG.csv"
f = open(csvFileName, 'w')
txt = "time, threshold 20%, threshold 25%, threshold 35%, dEGG, hybrid\n"
f.write(txt)
algoCount = len(arrDataCQ)
for i, t in enumerate(arrT):
	txt = "%f, " % arrT[i]
	for j in range(algoCount):
		txt += "%f" % arrDataCQ[j][i]
		if j < algoCount - 1:
			txt += ", "
		else:
			txt += "\n"
	f.write(txt)
f.close()



