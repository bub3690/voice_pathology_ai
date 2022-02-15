
"""
@package wavegram This module contains source code to analyze EGG signals with 
	the wavegram method. 

@copyright GNU Public License
@author written 2008-2016 by Christian Herbst (www.christian-herbst.org)

time
@note
How to cite the software:
Christian T. Herbst, W. T. S. Fitch, Jan G. Svec (2010). Electroglottographic wavegrams: a technique for visualizing vocal fold dynamics noninvasively. J. Acoust. Soc. Am., 128 (5), 3070-3078
Christian T. Herbst (2012). Investigation of glottal configurations in singing. Palacky University in Olomouc, the Czech Republic (Doctoral Dissertation)
@par
This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 3 of the License, or (at your option) any later 
version.
@par
This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
@par
You should have received a copy of the GNU General Public License along with 
this program; if not, see <http://www.gnu.org/licenses/>.

"""



######################################################################

import numpy
import math
import myWave
import wave
import pickle
import generalUtility
import dspUtil
import praatUtil
import matplotlib.pyplot as plt
import copy

CQ_DEGG = 1
CQ_THRESHOLD = 2
CQ_HYBRID = 3

######################################################################
# here be helper functions
######################################################################

def logger(msg):
	print ("\t\tWAVEGRAM: " + msg)

######################################################################

def corr(
		signal1, # input array
		signal2, # the other input array
		alignSize = True, # if true, the shorter array is scaled to the longer one
		zeroPaddingFactor = 2.0 # zero padding 
):
	MAX_FFT_LENGTH = 2**20
	
	signalSize1 = len(signal1)
	signalSize2 = len(signal2)
	if alignSize == False:
		if signalSize1 != signalSize2:
			raise Exception("the size of the two input signals does not match")
	
	signalSize = signalSize1
	if signalSize2 > signalSize: signalSize = signalSize2
	if signalSize <= 0 or signalSize >= MAX_FFT_LENGTH:
		raise Exception("the size of the input signals (" + str(signalSize) \
			+ ") is out of range.")
	
	# make the actual window size a power of two
	targetWindowSize = int(signalSize * float(zeroPaddingFactor))
	realWindowSize = 1
	while realWindowSize < targetWindowSize:
		realWindowSize *= 2
	
	if realWindowSize > MAX_FFT_LENGTH or realWindowSize < 1: 
		raise Exception("array size (" + str(realWindowSize) \
			+ ") is not allowed (range = 1 - " + str(MAX_FFT_LENGTH) + ").")
	
	arrIn1 = numpy.zeros(realWindowSize)
	arrIn2 = numpy.zeros(realWindowSize)
	arrIn3 = numpy.zeros(realWindowSize / 2)
	
	# copy the input data to the object's input data array
	# since the target array is longer, it is automatically zero-padded
	for k in range(signalSize1):
		arrIn1[k] = signal1[k]
	for k in range(signalSize2):
		arrIn2[k] = signal2[k]
	
	# calculate the FFTs
	arrFft1 = numpy.fft.rfft(arrIn1)
	arrFft2 = numpy.fft.rfft(arrIn2)
	
	# multiply result 1 with complex conjugate of result 2 and store it.
	for k in range(realWindowSize / 2):
		arrIn3[k] = arrFft1[k].conjugate() * arrFft2[k]
	
	# do the reverse fftp
	r = numpy.fft.irfft(arrIn3) * 2.0
	
	# favour smaller lags (avoid period doubling/tripling ...)
	for i in range(len(r)):
		r[i] *= 1.0 - i * 0.1 / len(r)
	
	# find the maximum in the lag function
	xOfMax, valMax = generalUtility.findArrayMaximum(r, int(round(signalSize * 0.25)), signalSize - 1)
	
	return xOfMax, valMax / r.max(), r

######################################################################

def rmsToDb(
		rmsValue, # the RMS value that should be converted to a dB value
		dbBase = 0, # the dB base value
		rmsBase = 1.0 # the RMS base value
	):
	""" performs a RMS to dB conversion """
	return dspUtil.rmsToDb(rmsValue, valueType = dspUtil.FIELD_QUANTITY, 
		dbBase = dbBase, rmsBase = rmsBase)
	
######################################################################
	

	



######################################################################
# array helper class
######################################################################

INTERPOLATE_NO = 0
INTERPOLATE_LINEAR = 1
INTERPOLATE_PARABOLIC = 2
INTERPOLATE_SINC = 3

######################################################################

class wavegramArray:
	""" This class wraps a numpy array. This design is necessary for two 
		reasons: To make sure that array data can be passed as both a list or 
		a numpy.array (and nothing else); and to implement a means for 
		parabolic and sinc interpolation (the latter being needed for 
		stretching arrays).
		TODO: change into a class derived from numpy.array
	"""
	
	def __init__(self, data):
		self.init(data)
		
	# ---------------------------------------------------------------------- #
		
	def getSize(self):
		return self.size	
		
	# ---------------------------------------------------------------------- #
		
	def init(self, data):
		objType = type(data).__name__.strip()
		if objType == "ndarray":
			self.data = data
			self.size = len(data)
		elif objType == "list":
			self.data = numpy.array(data)
			self.size = len(data)
		else:
			raise Exception('data argument is no instance of numpy.array')
		
	# ---------------------------------------------------------------------- #
			
	def get(self, x, method = INTERPOLATE_LINEAR, sincWidth = 30):
		if (x > self.size and x < self.size + 1): x = self.size
		if (x < 0 or x > self.size):
			raise Exception('x (' + str(x) + ') is out of bounds')
		
		if (x >= self.size - 1):
			return self.data[self.size - 1]

		left = int(x) # force to int
		right = left + 1
		pos = float(x) - left # force to floating point
		if (pos == 0.0):
			return self.data[left]
		
		if method == INTERPOLATE_NO:
			# no interpolation: index is truncated to integral number
			return self.data[left]
			
		if method == INTERPOLATE_LINEAR:
			# simple linear interpolation
			return self.data[left] * (1.0 - pos) + self.data[right] * pos
			
		if method == INTERPOLATE_PARABOLIC:
			# parabolic interpolations
			left_val = 0
			right_val = 0
			if (left > 0):
				# need tocalculate the left val
				left_val = interpolateParabolic(self.data[left - 1], \
					self.data[left], self.data[right], pos)
			else:
				left_val = generalUtility.interpolateLinear(self.data[0], self.data[1], pos)
			if (right < self.size - 1):
				# calculate the right val
				right_val = generalUtility.interpolateParabolic(self.data[left], \
					self.data[right], self.data[right + 1], pos - 1)
			else:
				right_val = self.data[self.size - 2] * (1.0 - pos) \
					+ self.data[self.size - 1] * pos;
			return left_val * (1.0 - pos) + right_val * pos;
		
		if method == INTERPOLATE_SINC:
			# sinc interpolation
			tmp = 0
			for i in range(sincWidth):
				if (left - i >= 0):
					# left-hand side
					tmp += self.data[left - i] * numpy.sinc(pos + i)
				if (right + i < self.size):
					# right-hand side
					tmp += self.data[right + i] * numpy.sinc(pos - (1.0 + i))
			return tmp
			
		raise Exception("unknown interpolation method")
		
	# ---------------------------------------------------------------------- #
		
	def stretch(self, newSize, method = INTERPOLATE_LINEAR, sincWidth = 30):
		newSize = int(newSize)
		dataNew = numpy.zeros(newSize)
		for i in range(newSize):
			idx = float(i) * (self.size + 1) / newSize
			dataNew[i] = self.get(idx, method, sincWidth)
		self.data = dataNew
		self.size = len(dataNew)



######################################################################
# signal and meta data
######################################################################

META_DATA_SUFFIX = '.meta'

class signalMetaData:
	
	def __init__(self):
		self.version = 1
		self.numFrames = 0
		self.fs = 0
		self.duration = 0
		self.audioChannel = 0
		self.eggChannel = -1
		self.f0ok = False
		self.F0progressPeriods = 0
		self.analysisTimeStep = 0
		self.arrF0 = { 't':[], 'freq':[] }
		self.arrSpl = []
		self.channels = 0
		self.glottalCyclesOk = False
		self.arrGlottalCycles = { 'offset':[], 'size':[] }
		pass

######################################################################

DETECT_GLOTTAL_CYCLES_CORR_IDEAL_CYCLE = 1
DETECT_GLOTTAL_CYCLES_ANALYTIC = 2
class signal:

	# TODO: I don't like the way the original WAVE data is currently stored. 
	#       change accordingly in a future release...

	def __init__(self, data = [], fs = 44100):
		self.init(data, fs)
		
	# ---------------------------------------------------------------------- #
	
	def init(self, data, fs):
		self.metaData = signalMetaData()
		# make sure data is of right type
		objType = type(data).__name__.strip()
		if objType == "ndarray":
			self.data = [data] # the data array is a numpy.array object
			self.metaData.numFrames = len(data)
		elif objType == "list":
			self.data = [numpy.array(data)] # the data array is a numpy.array object
			self.metaData.numFrames = len(data)
		else:
			raise Exception('data argument is no instance of numpy.array and no list')
		self.metaData.fs = fs
		self.metaData.duration = float(self.metaData.numFrames) / fs
		self.fileName = ''
		
	# ---------------------------------------------------------------------- #
		
	def getDuration(self):
		return self.metaData.duration
		
	# ---------------------------------------------------------------------- #
		
	def getRawData(self, channel):
		data = numpy.zeros(self.metaData.numFrames)
		for i in range(self.metaData.numFrames):
			data[i] = self.data[channel][i]
		return data
		
	# ---------------------------------------------------------------------- #
		
	def getSamplingFrequency(self):
		return self.metaData.fs
		
	# ---------------------------------------------------------------------- #
		
	def wrapWaveFile(self, fileName,
			forceCycleDetection = False,
			f0DetectionUsePraat = True,
			f0DetectionTargetChannel = 1,
			f0Min = 50,
			f0Max = 3000,
			f0DetectionVoicingThreshold = 0.3,
			f0DetectionApplyWindow = False,
			f0DetectionNumPeriods = 3,
			f0DetectionProgressPeriods = 1,
	        f0DetectionPraatTimeStep = None,
			f0DetectionPraatOctaveCost = 0.01,
			cycleDetectionEggFMax = 4000,
			cycleDetectionEggFMin = 5,
			cycleDetectionEggDataChannel = -1,
			cycleDetectionFilterBandwidth = 2,
			cycleDetectionMethod = DETECT_GLOTTAL_CYCLES_CORR_IDEAL_CYCLE,
			cycleDetectionTolerance = 0.0001,
			cycleDetectionNumPeriods = 1.8,
			cycleDetectionCorrCoeff = 0.2,
			cycleDetectionAdvance = 0.001,
			cycleDetectionNumRecentCycles = 0,
			verbose = False
	):
		"""
		:param fileName: the input WAV file, typically a two-channel file,
			containing the acoustic and electroglottographic (EGG) data
		:param forceCycleDetection: if True, we'll perform a F0 estimation and
			cycle detection even if a meta data file has been found
		:param f0DetectionUsePraat: if True, we'll use Praat to estimate the
			time-varying fundamental frequency. The praatUtil module must be
			available for this to work.
		:param f0DetectionTargetChannel: F0 detection based on channel N
			[0..numChannels-1] (i.e., zero-based)
		:param f0Min: don't look for f0 values below this limit [Hz]
		:param f0Max: don't look for F0 values above this limit [Hz]
		:param f0DetectionVoicingThreshold: autocorrelation-coefficient as
			criterion for F0 detection; works just like the "voicing threshold"
			in Praat
		:param f0DetectionApplyWindow: whether or not to window the data (Hann window)
		:param f0DetectionNumPeriods: window size = f0DetectionNumPeriods / fMin [s]
		:param f0DetectionProgressPeriods:
		:param f0DetectionPraatTimeStep: [s]; if None, we'll set this to
			1 / Fmax; only used if f0DetectionUsePraat is True
		:param f0DetectionPraatOctaveCost: the Praat 'octave cost' parameter
			(the higher, the less subharmonics will be considered)
		:param cycleDetectionEggFMax: low-pass filter EGG data at the frequency
		:param cycleDetectionEggFMin: high-pass filter EGG data at the frequency
		:param cycleDetectionEggDataChannel: zero-based. per default, use the
			last channel found in the WAV file
		:param cycleDetectionFilterBandwidth: Praat filter bandwidth
		:param cycleDetectionMethod: how to detect glottal cycles: either by
			cross-correlating them against an ideal waveform (typically an EGG
			waveform from Titze's kinematic model), or by turning the input
			data into an analytic signal and assessing the unwrapped phase.
			Allowed values for this parameter are:
				- DETECT_GLOTTAL_CYCLES_CORR_IDEAL_CYCLE
				- DETECT_GLOTTAL_CYCLES_ANALYTIC
		:param cycleDetectionTolerance: tolerance (ms) by which cycles can be offset
		:param cycleDetectionNumPeriods: range over which to look for cycles in
			cross-correlation with ideal waveform
		:param cycleDetectionCorrCoeff: only accept cycle if the cross-
			correlation maximum is above the given threshold
		:param cycleDetectionAdvance: seconds; advance search by this amount if
			no cycle has been found
		:param cycleDetectionNumRecentCycles: if greater than zero, we'll
			dynamically average the specified number of most recent glottal
			cycles in order to arrive at an "ideal glottal cycle" for
			consecutive cycle detection (this way the "ideal cycle" gets
			permanently updated, and we've essentially looking for similarity
			in consecutive cycles. If we don't have enough recent cycles,
			we'll substitute them with the "ideal cycle". THIS IS AN
			EXPERIMENTAL FEATURE, IT'S EXECUTION IS RATHER TIME-CONSUMING.
		:param verbose: if True, a few debugging messages are printed to stdout
		:return:
		"""
		if verbose: logger('wrapping file ' + fileName)
		self.fileName = fileName
		numChannels, numFrames, fs, arrChannelData = myWave.readWaveFile(fileName, 
			useRobustButSlowAlgorithm = False)
		try:
			# analysis data (such as F0 information) is stored in a meta data
			# file together with the analyzed WAVE file. try to locate it, in
			# order to avoid re-calculating the analysis data.
			self.loadMetaData()
			tmp = signalMetaData()
			msg = 'Unable to restore meta data for file ' + fileName + ': '
			if self.metaData.version != tmp.version:
				raise Exception(msg + 'meta data file version is outdated')
			if self.metaData.fs != fs:
				raise Exception(msg + 'sample rates do not match (changed?)')
			if self.metaData.channels != numChannels:
				raise Exception(msg + 'numChannels do not match (changed?)')
			if self.metaData.numFrames != numFrames:
				raise Exception(msg + 'numFrames do not match (changed?)')
			if forceCycleDetection:
				logger("forcing F0 calculation and cycle detection")
		except Exception as e:
			logger('WARNING: unable to load meta data: ' + e.__str__())
			self.metaData = signalMetaData()
			self.metaData.fs = fs
			self.metaData.channels = numChannels
			self.metaData.numFrames = numFrames
			self.saveMetaData()
		self.metaData.duration = float(self.metaData.numFrames) / self.metaData.fs
		self.data = arrChannelData

		# f0 detection
		if self.metaData.f0ok == False or forceCycleDetection:
			self.calculateF0(
				Fmin = f0Min,
				Fmax = f0Max,
			    numPeriods = f0DetectionNumPeriods,
				progressPeriods = f0DetectionProgressPeriods,
				targetChannel = f0DetectionTargetChannel,
				voicingThreshold = f0DetectionVoicingThreshold,
				applyWindow = f0DetectionApplyWindow,
				f0DetectionUsePraat = f0DetectionUsePraat,
				praatTimeStep = f0DetectionPraatTimeStep,
				praatOctaveCost = f0DetectionPraatOctaveCost,
				verbose=verbose
			)

		# glottal cycle detection
		if self.metaData.glottalCyclesOk == False or forceCycleDetection:
			self.detectGlottalCycles(
				eggFMax = cycleDetectionEggFMax,
				eggFMin = cycleDetectionEggFMin,
				filterBandwidth = cycleDetectionFilterBandwidth,
				method = cycleDetectionMethod,
				tolerance = cycleDetectionTolerance,
				numPeriods = cycleDetectionNumPeriods,
				corrCoeff = cycleDetectionCorrCoeff,
				advance = cycleDetectionAdvance,
				numRecentCycles = cycleDetectionNumRecentCycles,
				verbose = False
			)

	# ---------------------------------------------------------------------- #
		
	def calculateF0(self, 
		Fmin = 50,
		Fmax = 3000,
		numPeriods = 5.0,
		progressPeriods = 1,
		targetChannel = -1, # if -1, the EGG channel is used
		voicingThreshold = 0.3,
		applyWindow = False,
	    f0DetectionUsePraat = True, # if True, we'll use Praat to estimate the time-varying fundamental frequency
	    praatTimeStep = None, # [s]; if None, we'll set this to 1 / Fmax; only used if f0DetectionUsePraat is True
		praatOctaveCost = 0.1, # the Praat 'octave cost' parameter (the higher, the less subharmonics will be considered)
		verbose = False,
	):
		if verbose: logger('calculating F0 for file ' + self.fileName)
		if targetChannel == -1: targetChannel = self.metaData.eggChannel
		if targetChannel >= self.metaData.channels: targetChannel = 0

		if f0DetectionUsePraat:

			# Praat F0 detection
			import praatUtil
			if praatTimeStep is None:
				praatTimeStep = 1.0 / float(Fmax)
			arrT, arrF0 = praatUtil.calculateF0OfSignal(
				self.data[targetChannel],
				self.metaData.fs,
				tmpDataPath = None,
				readProgress = praatTimeStep,
				acFreqMin = Fmin,
				voicingThreshold = voicingThreshold,
				veryAccurate = False,
				fMax = Fmax,
				octaveJumpCost = 0.35,
				octaveCost = praatOctaveCost,
				voicedUnvoicedCost = 0.14,
				tStart = None,
				tEnd = None
			)
			self.metaData.arrF0['t'] = arrT
			self.metaData.arrF0['freq'] = arrF0

		else:

			# conventional F0 detection
			readSize = int(numPeriods * float(self.metaData.fs) / Fmin)
			offset = 0
			numCycles = 0
			self.metaData.arrF0 = { 't':[], 'freq':[] }
			self.metaData.F0progressPeriods = progressPeriods
			while (offset < self.metaData.numFrames):
				dataTmp = numpy.zeros(readSize)
				if self.metaData.channels == 1:
					for i in range(readSize):
						idx = int(i + offset - (readSize / 2))
						if idx >= 0 and idx < self.metaData.numFrames:
							dataTmp[i] = self.data[targetChannel][idx]
				else:
					for i in range(readSize):
						idx = int(i + offset - (readSize / 2))
						if idx < self.metaData.numFrames:
							dataTmp[i] = self.data[targetChannel][idx]

				# apply window
				if applyWindow:
					fftWindow = createLookupTable(len(dataTmp), \
						LOOKUP_TABLE_HANN)
					dataTmp *= fftWindow

				# autocorrelation
				result = numpy.correlate(dataTmp, dataTmp, mode = 'full', \
					old_behavior = False)
				r = result[int(round(result.size/2)):] / readSize

				# find peak in AC
				xOfMax, valMax = generalUtility.findArrayMaximum(r, self.metaData.fs / Fmax, \
					self.metaData.fs / Fmin)
				valMax /= max(r)
				freq = self.metaData.fs / xOfMax
				periodSize = 0
				if freq > 0:
					periodSize = self.metaData.fs / freq
				t = (offset + (periodSize / 2.0))/self.metaData.fs
				if freq >= Fmin and freq <= Fmax and valMax >= voicingThreshold:
					self.metaData.arrF0['t'].append(t)
					self.metaData.arrF0['freq'].append(freq)
				else:
					# set F0 to zero if out of bounds
					self.metaData.arrF0['t'].append(t)
					self.metaData.arrF0['freq'].append(0)

				if periodSize > 10:
					offset += periodSize * progressPeriods
				else:
					offset += 10


		self.metaData.f0ok = True
		self.saveMetaData()
		
	# ---------------------------------------------------------------------- #

	def detectGlottalCycles(self, 
			eggFMax = None, # low-pass filter the EGG data at this freq. if None, we won't apply a filter
			eggFMin = None, # high-pass filter the EGG data at this freq. if None, we won't apply a filter
			filterBandwidth = 2, # Praat filter bandwidth
			method = DETECT_GLOTTAL_CYCLES_CORR_IDEAL_CYCLE,
			tolerance = 0, # tolerance (ms) by which cycles can be offset
			numPeriods = 1.8, # range over which to look for cycles in cross-
							# correlation with ideal waveform
			corrCoeff = 0.2, # only accept cycle if cross-correlation max is 
							 # above the given threshold
			advance = 0.005, # seconds; advance search by this amount if no 
							# cycle has been found
	        numRecentCycles = 0, # if greater than zero, we'll dynamically
	                        # average the specified number of most recent
	                        # glottal cycles in order to arrive at an "ideal
	                        # glottal cycle" for consecutive cycle detection
	                        # (this way the "ideal cycle" gets permanently
	                        # updated, and we've essentially looking for
	                        # similarity in consecutive cycles. If we don't have
	                        # enough "recent cycles", we'll fill that data with
	                        # the "ideal cycle".
			verbose = False,
	):
	
		if method == DETECT_GLOTTAL_CYCLES_ANALYTIC:
			# 2014-12-03: added new analytic signal algo
			if verbose: logger('detecting glottal cycles')
			arrEggLowpass = None
			if (not eggFMax is None) and (not eggFMin is None):
				arrEggLowpass = praatUtil.applyBandPassFilterToSignal(
					self.data[self.metaData.eggChannel], 
					self.metaData.fs, eggFMin, eggFMax, filterBandwidth,
					preservePhase = True)
			else:
				arrEggLowpass = copy.deepcopy(self.data[self.metaData.eggChannel])
			
			arrCycles = dspUtil.detectCyclesAnalytic(arrEggLowpass, 
				minDistanceConsecutiveCycles = self.metaData.fs / eggFMax)
			n = len(arrCycles)
			arrOffset, arrSize, arrF0 = [], [], []
			for i in range(1, n):
				offset = arrCycles[i-1]
				size = arrCycles[i] - offset
				f0 = float(size) / float(self.metaData.fs)
				arrOffset.append(offset)
				arrSize.append(size)
				arrF0.append(f0)
			
			self.metaData.arrGlottalCycles = { 'offset':arrOffset, 'size':arrSize, \
				'F0':arrF0 }
			self.metaData.glottalCyclesOk = True
			self.saveMetaData()
		
		
		elif method == DETECT_GLOTTAL_CYCLES_CORR_IDEAL_CYCLE:
			if self.metaData.f0ok == False:
				self.calculateF0()
			self.arrGlottalCycles = { 'offset':[], 'size':[] }
			arrOffset = []
			arrSize = []
			arrF0 = []
			dataSize = len(self.data[0])
			eggChannel = self.metaData.eggChannel
			recentOffset = 0


			self.arrGlottalCyclesTmp = []
			for i in range(len(self.metaData.arrF0['t']) - 1):
				t1 = self.metaData.arrF0['t'][i]
				freq1 = self.metaData.arrF0['freq'][i]
				t2 = self.metaData.arrF0['t'][i + 1]
				freq2 = self.metaData.arrF0['freq'][i + 1]
				if freq1 > 0 and freq2 > 0:
				
					t = float(t1)
					while t < t2:
						#try:
						if 1 == 1:
							tRel = (t - t1) / (t2 - t1)
							freq = generalUtility.interpolateLinear(freq1, freq2, tRel)
							offset = int(t * float(self.metaData.fs))
							size = int(self.metaData.fs / freq)
							idealWaveform = self.getIdealEggWaveform(size, numRecentCycles)
							if 1 == 2:
								plt.close()
								plt.plot(idealWaveform)
								plt.show()
								if len(self.arrGlottalCyclesTmp) > 20: exit(1)
							offset = self.metaData.fs * t
							left = int(offset)
							right = int(offset + size * numPeriods)
							if left < 0: left = 0
							if right >= dataSize: right = dataSize - 1
							if right > left:
								dataTmp = numpy.zeros(right - left)
								for k in range(int(right - left)):
										dataTmp[k] = self.data[eggChannel][left + k]
								valMax = max(dataTmp)
								valMin = min(dataTmp)
								dataTmp -= valMin
								dataTmp /= (valMax - valMin)
								xOfMax = -1
								try:
									xOfMax, yOfMax, r = corr(dataTmp, idealWaveform)
								except:
									pass
								if xOfMax > 0 and yOfMax >= corrCoeff:
									xOfMax -= size * 0.1 # shift cycle to the left
									cycleOffset = xOfMax + left
									#print t, cycleOffset, recentOffset
									if cycleOffset > recentOffset:
										#print "\t", t, xOfMax
										arrOffset.append(cycleOffset)
										arrSize.append(size)
										arrF0.append(freq)

										if 1 == 2:
											print(yOfMax, corrCoeff)
											plt.close()
											plt.clf()
											plt.plot(dataTmp)
											plt.plot(idealWaveform)
											plt.show()
											if len(self.arrGlottalCyclesTmp) > 20: exit(1)

										# remember recent cycles ...
										self.arrGlottalCyclesTmp.append(dataTmp)

										recentOffset = int(cycleOffset + size * 0.7)
										t = (recentOffset) / float(self.metaData.fs)
									else:
										t += advance
								else:
									if verbose:
										logger("WARNING: negative xOfMax value (" \
											+ str(xOfMax) + ") at t = " + str(t))
									t += advance
							else:
								t += advance
						#except Exception as e:
						#	print e
						#	t += advance
	
					
			# convert offsets and sizes to integers
			numCycles = len(arrOffset)
			for i in range(numCycles):
				arrOffset[i] = int(round(arrOffset[i]))
				arrSize[i] = int(round(arrSize[i]))
					
			# clean cases where we have a cycle overlap, or a little gap between
			# cycles
			toleranceSamples = int(round(((tolerance / 1000.0) * self.metaData.fs)))
			for i in range(numCycles - 1):
				offset1 = arrOffset[i]
				offset2 = arrOffset[i + 1]
				size1 = arrSize[i]
				size2 = arrSize[i + 1]
				gap = offset2 - (offset1 + size1)
				if gap != 0 and abs(gap) <= toleranceSamples:
					# gap/offset is within tolerance limits. fix it
					correction = gap / 2
					arrSize[i] += correction
					arrOffset[i + 1] -= gap - correction
					arrSize[i + 1] += gap - correction		
			
			self.metaData.arrGlottalCycles = { 'offset':arrOffset, 'size':arrSize, \
				'F0':arrF0 }
			self.metaData.glottalCyclesOk = True
			self.saveMetaData()
		
	# ---------------------------------------------------------------------- #
				
	def __getGlottalCycleRecursive(self, t, idxMin, idxMax, idxRecent, \
			tolerance, iteration
	):
		"""
		private method used to calculate glottal cycles
		@param t:
		@param idxMin:
		@param idxMax:
		@param idxRecent:
		@param tolerance:
		@param iteration:
		@return:
		"""
		if idxMax < idxMin:
			return -1
		idx = int((idxMax + idxMin) / 2.0)
		if idx == idxRecent: 
			return -1
		offset = float(self.metaData.arrGlottalCycles['offset'][idx]) \
			/ float(self.metaData.fs)
		size = self.metaData.arrGlottalCycles['size'][idx]
		duration = size / float(self.metaData.fs)
		if t + tolerance >= offset and t - tolerance <= (offset + duration):
			# found a cycle
			return idx
		if t < offset:
			# look in first half of array...
			return self.__getGlottalCycleRecursive(t, idxMin, idx, idx, \
				tolerance, iteration + 1)
		else:
			# look in second half of array...
			return self.__getGlottalCycleRecursive(t, idx, idxMax, idx, \
				tolerance, iteration + 1)
		
	# ---------------------------------------------------------------------- #

	def getNumCycles(self):
		return len(self.metaData.arrGlottalCycles['size'])

	# ---------------------------------------------------------------------- #
				
	def getGlottalCycleByIdx(self, idx):
		n = self.metaData.arrGlottalCycles['size'][idx]
		offset = self.metaData.arrGlottalCycles['offset'][idx]
		cycleData = self.data[self.metaData.eggChannel][offset:offset+n]
		return copy.deepcopy(cycleData), offset
		
	# ---------------------------------------------------------------------- #
				
	def getGlottalCycle(
			self, t, derivative = False, 
			tolerance = 0.00, normalize = True
	):
		if t < 0 or t > self.metaData.duration:
			return None #glottalCycle(numpy.array([1]), self.metaData.fs)
		fs = float(self.metaData.fs)
		candidate = self.__getGlottalCycleRecursive(t, 0, \
			len(self.metaData.arrGlottalCycles['offset']) - 1, -1, tolerance, 1)
		#print t, candidate
		if candidate != -1:
			try:
				size = int(self.metaData.arrGlottalCycles['size'][candidate])
				offset = int(self.metaData.arrGlottalCycles['offset'][candidate])
				dataTmp = numpy.zeros(size)
				for j in range(size):
						dataTmp[j] = self.data[self.metaData.eggChannel][j + offset]
				if derivative:
					dataTmp2 = numpy.copy(dataTmp)
					for j in range(size - 1):
						dataTmp[j] = dataTmp2[j + 1] - dataTmp2[j]
					dataTmp[size - 1] = 0
					valMin = min(dataTmp)
					valMin *= -1.0
					valMax = max(dataTmp)
					for j in range(size):
						val = dataTmp[j]
						if val > 0:
							val /= valMax
						else:
							val /= valMin
						dataTmp[j] = val
				return CGlottalCycle(dataTmp, self.metaData.fs, \
					normalize = normalize, offset = offset)
			except Exception as e:
				print ("WARNING:", e)
				return None
			
		return None #glottalCycle(numpy.zeros(1), self.metaData.fs)
		#raise Exception('no data found')
		
	# ---------------------------------------------------------------------- #
		
	def createWavegram(
			self, 
			width = 600, # resolution on x axis
			height = 300, # resolution on y axis 
			derivative = True, 
			tStart = 0, 
			tEnd = -1, 
			doColorize = True, 
			eggDbMin = 0, 
			eggDbMax = 0, 
			amplitudeData = None, 
			col1 = 0xFF0000, 
			col2 = 0x0000FF, 
			verbose = False,
			cycleDetectionTolerance = 0.0001,
			maxRecentCycles = 1, # use n recent cycles if no cycle found at t
			amplitudeScalingPower = 1, # non-linear scaling of normalized
				# intra-cycle amplitude: y = x ^ amplitudeScalingPower
				# default = 1 (no scaling)
	):
		if verbose: logger('creating wavegram')
		if tEnd == -1: tEnd = self.metaData.duration
		arraySize = width * height * 3
		tmp = numpy.zeros(arraySize)
		data = tmp.reshape(height,width,3) # y, x, RGB
		if doColorize and not amplitudeData:
			amplitudeData = self.calculateRmsVector(0.1, self.metaData.eggChannel, 0.02, True)
		if eggDbMin == 0 and eggDbMax == 0:
			eggDbMin = amplitudeData[1].min()
			eggDbMax = amplitudeData[1].max()
		recentCycle = None
		recentCycleCount = 0
		for x in range(width):
			t = tStart + x * (tEnd - tStart) / float(width - 1)
			glottalCycle = self.getGlottalCycle(t, derivative = derivative, \
				tolerance = cycleDetectionTolerance)
			if not glottalCycle:
				# if no glottal cycle was found, allow to display the most
				# recent cycle (but only once for consecutive missing cycles)
				if recentCycleCount < maxRecentCycles:
					glottalCycle = recentCycle
				recentCycleCount += 1
			else:
				recentCycleCount = 0
			recentCycle = glottalCycle
			amp = 0
			found = False
			for i in range(len(amplitudeData[0])):
				if amplitudeData[0][i] >= t:
					if i == 0: amp = amplitudeData[1][i]
					elif i >= len(amplitudeData[0]): amp = amplitudeData[1][-1]
					else:
						weighting = float(t - amplitudeData[0][i - 1])
						weighting /= float(amplitudeData[0][i] - \
							amplitudeData[0][i - 1])
						amp = generalUtility.interpolateLinear(amplitudeData[1][i - 1], \
							amplitudeData[1][i], weighting)
					found = True
					break
			if not found:
				try:
					if t < amplitudeData[0][0]:
						amp = amplitudeData[1][0]
					else:
						amp = amplitudeData[1][-1]
				except:
					amp = numpy.NaN
			ampScaling = (amp - eggDbMin) / (eggDbMax - eggDbMin)
			#print x, amp, ampScaling
			redFg = (col1 & 0xFF0000) * ampScaling + (col2 & 0xFF0000) \
				* (1.0 - ampScaling)
			greenFg = (col1 & 0x00FF00) * ampScaling + (col2 & 0x00FF00) \
				* (1.0 - ampScaling)
			blueFg = (col1 & 0x0000FF) * ampScaling + (col2 & 0x0000FF) \
				* (1.0 - ampScaling)
			redFg /= (65536 * 256.0)
			greenFg /= (256 * 256.0)
			blueFg /= 256.0
			redBg = 1.0
			greenBg = 1.0
			blueBg = 1.0
			#print x, amp, ampScaling, redFg, greenFg, blueFg
			if glottalCycle:
				strip = wavegramStrip(glottalCycle, height)
				if amplitudeScalingPower != 1:
					strip.scaleQuadratic(amplitudeScalingPower)
				for y in range(height):
					if glottalCycle.getSize() <= 1:
						# draw white if invalid glottal cycle
						for i in range(3):
							data[y][x][i] = 1
					else:
						if doColorize:
							val = strip.data[y]
							r = redBg * (1.0 - val) + redFg * val
							b = blueBg * (1.0 - val) + greenFg * val
							g = greenBg * (1.0 - val) + blueFg * val
							# check for clipping:
							if r > 1: r = 1
							if r < 0: r = 0
							if b > 1: b = 1
							if b < 0: b = 0
							if g > 1: g = 1
							if g < 0: g = 0
							data[y][x][0] = r
							data[y][x][1] = b
							data[y][x][2] = g
						else:
							col = 1.0 - strip.data[y]
							for i in range(3):
								data[y][x][i] = col
			else:
				# draw white if invalid glottal cycle
				for y in range(height):
					for i in range(3):
						data[y][x][i] = 1
		return data
		
	# ---------------------------------------------------------------------- #
		
	def saveMetaData(self):
		f = open(self.fileName + META_DATA_SUFFIX, "w") 
		pickle.dump(self.metaData, f)
		f.close()
		
	# ---------------------------------------------------------------------- #

	def loadMetaData(self):
		f = open(self.fileName + META_DATA_SUFFIX, "r")
		self.metaData = pickle.load(f)
		f.close()
		
	# ---------------------------------------------------------------------- #
	
	def calculateRmsVector(self, 
			windowSize, # seconds
			channel,
			timeStep, # seconds
			convertToDb = False
		):
		if timeStep <= 0:
			raise Exception("time step must be greater than zero")
		arrDataX = []
		arrDataY = []
		offset = 0
		while offset < self.metaData.duration:
			arrDataX.append(offset)
			tStart = offset - windowSize / 2.0
			if tStart < 0: tStart = 0
			tEnd = offset + windowSize / 2.0
			if tEnd > self.metaData.duration: tEnd = self.metaData.duration
			val = -99
			#try:
			if 1 == 1:
				val = self.calculateRmsScalar(channel, tStart, tEnd, convertToDb)
			#except Exception as e: 
			#	print "\t\tWARNING calculateRmsVector(...):", e 
			arrDataY.append(val)
			offset += timeStep
		return numpy.array(arrDataX), numpy.array(arrDataY)
		
	# ---------------------------------------------------------------------- #
				
	def calculateRmsScalar(self, channel, tStart = 0, tEnd = 0, 
			convertToDb = False
	):
		if tEnd == 0: tEnd = self.metaData.duration
		if tStart < 0:
			raise Exception("tStart must not be smaller than zero")
		if tEnd > self.metaData.duration:
			raise Exception("tEnd out of range")
		if (tEnd <= tStart):
			raise Exception("tEnd must be greater than tStart")
			
		#print channel, tStart, tEnd, convertToDb	
		sum = 0
		offset1 = int(tStart * float(self.metaData.fs))
		offset2 = int(tEnd * float(self.metaData.fs))
		if offset1 < 0: offset1 = 0
		if offset2 >= self.metaData.numFrames: 
			offset2 = self.metaData.numFrames - 1
		numSamples = offset2 - offset1
		#print offset1, offset2, channel, len(self.data[channel])
		for i in range(numSamples):
			sum += self.data[channel][i + offset1]
		mean = sum / float(numSamples)
		tmp = 0
		for i in range(numSamples):
			tmp2 = self.data[channel][i + offset1] - mean
			tmp += (tmp2 * tmp2)
		tmp /= float(numSamples + 1)
		val = math.sqrt(tmp)
		if convertToDb:
			if val == 0:
				val = -99
			else:
				val = rmsToDb(val)
		return val	
		
	# ---------------------------------------------------------------------- #
		
	def calculateCQ(self, method = CQ_DEGG, threshold = 0.2, doInterpolate = True):
		"""
		calculate the EGG contact quotient of this glottal cycle
		@param method one of these: CQ_DEGG (positive and negative DEGG peak),
			CQ_THRESHOLD (threshold-based), CQ_HYBRID (positive DEGG peak
			for contacting 3/7 threshold for de-contacting
		@param threshold threshold percentage [0.1..0.9], only applied when 
			method = CQ_THRESHOLD
		@param doInterpolate if True, we'll interpolate the results to squeeze
			out a bit more accuracy
		@return a list containing the time offsets of all cycles, the 
			respective calculated CQ (None if unable to calculate), and the 
			respective contacting and de-contacting offsets
		"""
		n = self.getNumCycles()
		arrT = [None] * n
		arrCQ = [None] * n
		arrContactingOffset = [None] * n
		arrDecontactingOffset = [None] * n
		for idx in range(n):
			cycleData, offset = self.getGlottalCycleByIdx(idx)
			cycle = CGlottalCycle(cycleData, self.getSamplingFrequency(),
				normalize = True, offset = offset)
			cq, o1, o2 = cycle.calculateCQ(method, threshold, doInterpolate)
			arrT[idx] = float(offset) / float(self.getSamplingFrequency())
			arrCQ[idx] = cq
			arrContactingOffset[idx] = o1
			arrDecontactingOffset[idx] = o2
		return arrT, arrCQ, arrContactingOffset, arrDecontactingOffset

	# ---------------------------------------------------------------------- #

	def getIdealEggWaveform(self, size, numRecentCycles = 0):
		"""
		@param size: the length [frames] of the cycle that is returned
		@param detectGlottalCycles: if numRecentCycles is greater than zero, we'll
			consider recent cycles for generating the returned cycle. see the
			description of the numRecentCycles parameter in the method
			detectGlottalCycles
		@return: a numpy array
		"""
		tmp = [ 0.016129032, 0.118951613, 0.231854839, 0.336693548, 0.435483871,
            0.524193548, 0.60483871, 0.675403226, 0.737903226, 0.790322581,
            0.842741935, 0.885080645, 0.927419355, 0.953629032, 0.967741935,
            0.977822581, 0.987903226, 0.991935484, 0.997983871, 1, 1,
            0.997983871, 0.995967742, 0.989919355, 0.985887097, 0.977822581,
            0.965725806, 0.955645161, 0.945564516, 0.933467742, 0.921370968,
            0.909274194, 0.89516129, 0.875, 0.856854839, 0.838709677,
            0.818548387, 0.798387097, 0.778225806, 0.756048387, 0.731854839,
            0.705645161, 0.675403226, 0.64516129, 0.608870968, 0.570564516,
            0.528225806, 0.483870968, 0.433467742, 0.370967742, 0.304435484,
            0.213709677, 0.092741935, 0.008064516, 0.008064516, 0.008064516,
            0.006048387, 0.006048387, 0.006048387, 0.006048387, 0.004032258,
            0.004032258, 0.004032258, 0.004032258, 0.004032258, 0.004032258,
            0.004032258, 0.004032258, 0.004032258, 0.004032258, 0.004032258,
            0.002016129, 0.002016129, 0.002016129, 0.002016129, 0.002016129,
            0.002016129, 0.002016129, 0.002016129, 0.002016129, 0.002016129,
            0.002016129, 0.002016129, 0.002016129, 0.002016129, 0.002016129,
            0.002016129, 0.002016129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002016129,
            0.002016129 ]
		# tmp = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	     #    0.136383057, 0.264526367, 0.382476807, 0.489624023, 0.584075928,
	     #    0.670227051, 0.741760254, 0.794250488, 0.835571289, 0.869598389,
	     #    0.897247314, 0.919189453, 0.940185547, 0.956726074, 0.968780518,
	     #    0.97769165, 0.988189697, 0.992004395, 0.997741699, 0.998687744,
	     #    0.999969482, 0.998352051, 0.995513916, 0.99230957, 0.987884521,
	     #    0.979919434, 0.972290039, 0.962768555, 0.951934814, 0.939849854,
	     #    0.927154541, 0.914093018, 0.898193359, 0.88104248, 0.863555908,
	     #    0.843841553, 0.825408936, 0.804412842, 0.782775879, 0.760528564,
	     #    0.735412598, 0.709350586, 0.682647705, 0.652099609, 0.624145508,
	     #    0.592956543, 0.55960083, 0.523651123, 0.487091064, 0.447021484,
	     #    0.408233643, 0.364349365, 0.316986084, 0.26739502, 0.216522217,
	     #    0.163085938, 0.105560303, 0.0498962402, 0.00476074219, 0, 0, 0, 0,
	     #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	     #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

		#data = wavegramArray(tmp)
		#data.stretch(size)

		idealCycle = numpy.array(dspUtil.stretchArray(tmp, size))

		if numRecentCycles > 0:
			data = numpy.zeros(size)
			for i, recentCycle in enumerate(self.arrGlottalCyclesTmp):
				data += dspUtil.stretchArray(recentCycle, size)
			m = len(self.arrGlottalCyclesTmp)
			for i in range(numRecentCycles - m):
				data += idealCycle
			data /= float(numRecentCycles)
			return data

		return idealCycle

######################################################################



######################################################################
# glottal cycle
######################################################################

class CGlottalCycle:
	""" this class represents sample data for one glottal cycle """
	
	def __init__(self, data, samplingRate, normalize = True, offset = None,
	        f0 = None):
		""" initialization of a CGlottalCycle object. unless you explicitly
			pass some data, this function is no substitute for the init(...) 
			function
		"""
		self.init(data, samplingRate, normalize = normalize, offset = offset,
		    f0 = f0)
		
	# ---------------------------------------------------------------------- #

	def init(self, data, samplingRate, normalize = True, offset = None,
	        f0 = None):
		"""
		Initialize the CGlottalCycle object with some meaningful data
		@param data: a numpy array (or a list of floats), containing the
			cycle data
		@param samplingRate: sampling frequency [Hz]
		@param normalize: if True, we'll normalize the glottal cycle data
		@param offset: the extraction offset [samples] within the analyzed
			EGG signal
		@param f0: optional f0 information [Hz]. Note that this f0 information
			may differ from dividing the sampling frequency by the glottal
			cycle length (given in samples).
		@return:
		"""

		# make sure data is of right type
		objType = type(data).__name__.strip()
		self.__offset = offset
		if objType == "ndarray":
			self.__data = wavegramArray(copy.deepcopy(data)) # cast to wavegram.wavegramArray
			self.__size = len(data)
		elif objType == "list":
			self.__data = wavegramArray(copy.deepcopy(data)) # cast to wavegram.wavegramArray
			self.__size = len(data)
		else:
			raise Exception('data argument is no instance of numpy.array')

		# set member variables
		self.__samplingFreq = samplingRate
		self.__f0 = f0

		#normalize
		valMax = self.__data.data.max()
		valMin = self.__data.data.min()
		for i in range(self.__size):
			if normalize:
				self.__data.data[i] = (self.__data.data[i] - valMin) \
					/ (valMax - valMin)

	# ---------------------------------------------------------------------- #

	def getSize(self):
		return self.__data.getSize()
		
	# ---------------------------------------------------------------------- #
		
	def getOffset(self):
		return self.__offset
		
	# ---------------------------------------------------------------------- #

	def getWavegramArray(self):
		return self.__data

	# ---------------------------------------------------------------------- #
		
	def getData(self):
		return self.__data.data
		
	# ---------------------------------------------------------------------- #

	def getCycleF0(self):
		T = self.__size / float(self.__samplingFreq)
		return 1.0 / T

	# ---------------------------------------------------------------------- #

	def getOptionalF0(self):
		return self.__f0

	# ---------------------------------------------------------------------- #

	def getDiffF0Percent(self):
		if self.__f0 is None:
			raise Exception("optional f0 parameter has never been set")
		return 100.0 * abs(self.__f0 - self.getCycleF0()) / float(self.__f0)

	# ---------------------------------------------------------------------- #
	
	def getSamplingFrequency(self):
		return self.__samplingFreq

	# ---------------------------------------------------------------------- #
		
	def calculateCQ(self, method = CQ_DEGG, threshold = 0.2, doInterpolate = True):
		"""
		calculate the EGG contact quotient of this glottal cycle
		@param method one of these: CQ_DEGG (positive and negative DEGG peak),
			CQ_THRESHOLD (threshold-based), CQ_HYBRID (positive DEGG peak
			for contacting, typically a 3/7 threshold for de-contacting)
		@param threshold threshold percentage [0.1..0.9], only applied when 
			method = CQ_THRESHOLD
		@param doInterpolate if True, we'll interpolate the results to squeeze
			out a bit more accuracy
		@return a list containing the calculated CQ (None if unable to
			calculate, the contacting offset and the de-contacting offset)
		"""
		data = self.__data.data
		n = len(data)
		o1 = numpy.nan # contacting offset
		o2 = numpy.nan # de-contacting offset
		if method == CQ_DEGG:
			drv = dspUtil.toDerivative(data)
			xOfMax, valMax = generalUtility.findArrayMaximum(drv, 
				doInterpolate = doInterpolate)
			o1 = xOfMax
			oTmp = int(round(o1))
			try:
				xOfMax, valMax = generalUtility.findArrayMaximum(drv[oTmp:] * -1.0,
					doInterpolate = doInterpolate)
				o2 = xOfMax + oTmp
			except Exception as e:
				print("WARNING:", str(e))
		elif method == CQ_THRESHOLD:
			tmp = dspUtil.normalize(data, maxOut = 1.0, minOut = 0.0)
			if threshold < 0.1 or threshold > 0.9:
				raise Exception("threshold out of range")
			for i in range(n):
				if tmp[i] >= threshold:
					if doInterpolate and i > 0:
						y1 = tmp[i-1]
						y2 = tmp[i]
						o1 = i + (threshold - y1) / float(y2 - y1) - 1
						# y1 = tmp[i] - threshold
						# y2 = threshold - tmp[i-1]
						# o1 = i + y2 / float(y1+y2)
					else:
						o1 = i
					break
			if o1:
				for i in range(int(o1)+1, n):
					if tmp[i] < threshold:
						if doInterpolate and i > 0:
							y1 = tmp[i-1]
							y2 = tmp[i]
							o2 = i + (threshold - y1) / float(y2 - y1) - 1
							# y1 = tmp[i] - threshold
							# y2 = threshold - tmp[i-1]
							# o2 = i + y2 / float(y1+y2)
						else:
							o2 = i
						break
		elif method == CQ_HYBRID:
			drv = dspUtil.toDerivative(data)
			xOfMax, valMax = generalUtility.findArrayMaximum(drv,
				doInterpolate = doInterpolate)
			o1 = xOfMax
			tmp = dspUtil.normalize(data, maxOut = 1.0, minOut = 0.0)
			if o1:
				for i in range(int(o1)+1, n):
					if tmp[i] < threshold:
						if doInterpolate and i > 0:
							y1 = tmp[i-1]
							y2 = tmp[i]
							o2 = i + (threshold - y1) / float(y2 - y1) - 1
							# y1 = tmp[i] - threshold
							# y2 = threshold - tmp[i-1]
							# o2 = i + y2 / float(y1+y2)
						else:
							o2 = i
						break
		else:
			raise Exception("invalid method specified")
			
		if 1 == 2:
			plt.clf()
			plt.plot(data)
			plt.plot([o1, o1], [0, 1])
			plt.plot([o2, o2], [0, 1])	
			plt.grid()
			plt.show()
			
		if o1 and o2 and o2 > o1:
			cq = float(o2-o1) / float(n)
			return cq, o1, o2
		return None, o1, o2
		
	# ---------------------------------------------------------------------- #

	def calculateFFT(self, spectrumType = dspUtil.AMPLITUDE_SPECTRUM,
	        convertToDb = False):
		"""
		calculate the DTFT of the cycle data, i.e., each spectrum coefficient
		represents one harmonic
		:param spectrumType: dspUtil.AMPLITUDE_SPECTRUM or dspUtil.POWER_SPECTRUM
		:param convertToDb: either True or False; influences the output
		:return: the FFT coefficients
		"""
		specX, specY = dspUtil.calculateFFT(self.getData(),
		    self.getSamplingFrequency(), self.getSize(),
			applyWindow = False, convertToDb = convertToDb,
			spectrumType = spectrumType, zeroPaddingFactor = 1)
		return specX, specY

	# ---------------------------------------------------------------------- #

	def calculateSpectralTilt(self, numHarmonics = 10, arrDtftFreq = None,
	        arrDtftAmp = None, includeDC = False):
		"""
		calculate the spectral tilt, specified in (negative) dB per octave, via
		a linear fit through the amplitudes of the freq-log spectrum
		:param numHarmonics: the number of harmonics that should be considered
		:param arrDtftFreq: a list with the frequencies of the DTFT spectrum,
			including the DC offset. if None, we'll ask for this data by calling
			calculateFFT(...)
		:param arrDtftAmp: a list with the amplitudes of the DTFT spectrum,
			including the DC offset. if None, we'll ask for this data by calling
			calculateFFT(...)
		:param includeDC: If True (not recommended), the DC offset is included
			in spectral slope calculation
		:return: the spectral slope (or spectral tilt), specified in (negative)
			dB per octave
		"""
		if arrDtftAmp is None or arrDtftFreq is None:
			arrDtftFreq, arrDtftAmp = self.calculateFFT()
		idx1 = 1
		if includeDC: idx1 = 0
		idx2 = idx1 + numHarmonics
		arrXtmp = numpy.zeros(numHarmonics)
		arrYtmp = numpy.zeros(numHarmonics)
		for k in range(idx1, idx2):
			arrXtmp[k - idx1] = dspUtil.hertzToCents(arrDtftFreq[k]) / 1200.0
			arrYtmp[k - idx1] = arrDtftAmp[k]
		slope, offset = numpy.polyfit(arrXtmp, arrYtmp, 1)
		return slope

	# ---------------------------------------------------------------------- #

	def calculateNAQ(self):
		"""
		calculate the NAQ parameter, as described in Eq. 1 of:
		Enflo, L., Herbst, C. T., Sundberg, J., and McAllister, A. (2016).
		"Comparing Vocal Fold Contact Criteria Derived From Audio and
		Electroglottographic Signals," J. Voice, 30, 381-388.
		doi:10.1016/j.jvoice.2015.05.015
		:return:
		"""
		data = self.getData()
		drv = dspUtil.toDerivative(data)
		NAQ = (numpy.nanmax(data) - numpy.nanmin(data)) \
		      / (float(len(data)) * numpy.nanmax(drv))
		return NAQ

	# ---------------------------------------------------------------------- #

	def calculateINAQ(self, NAQ = None):
		"""
		calculate the iNAQ parameter, as described in Eq. 1 of:
		Enflo, L., Herbst, C. T., Sundberg, J., and McAllister, A. (2016).
		"Comparing Vocal Fold Contact Criteria Derived From Audio and
		Electroglottographic Signals," J. Voice, 30, 381-388.
		doi:10.1016/j.jvoice.2015.05.015
		:return:
		"""
		if NAQ is None:
			NAQ = self.calculateNAQ()
		iNAQ = 1.0 / (numpy.pi * NAQ)
		return iNAQ

	# ---------------------------------------------------------------------- #

	def calculateRelativeContactRiseTime(self, lowerThreshold = 0.2, upperThreshold = 0.8):
		"""
		calculate the relative contact rise time of the cycle
		@param lowerThreshold:
		@param upperThreshold:
		@return:
		"""
		data = dspUtil.normalize(self.getData())
		n = len(data)
		o1 = -1
		o2 = -1
		for i in range(n):
			if data[i] >= lowerThreshold:
				o1 = i
				break
		for i in range(o1, n):
			if data[i] >= upperThreshold:
				o2 = i
				break
		RCRT = float(o2 - o1) / float(n)
		return RCRT


	# ---------------------------------------------------------------------- #
		


######################################################################
# a wavegram "strip"
######################################################################

class wavegramStrip:
	""" this class represents one "strip" in the wavegram. a strip can be
		created by passing an object of type CGlottalCycle to the init function
	"""
	
	def __init__(
			self, 
			data, 
			size, 
			normalize = True # if False, a non-normalized version is created
	):
		self.init(data, size, normalize)
		
	# ---------------------------------------------------------------------- #
		
	def init(
			self,
			glottalCycleData,
			size,
			normalize = True # if False, a non-normalized version is created
	):
		
		if isinstance(glottalCycleData, CGlottalCycle):
			if normalize:
				dataTmp = glottalCycleData.getWavegramArray()
				dataTmp.stretch(size)
				self.data = dataTmp.data	
			else:
				self.data = glottalCycleData.getWavegramArray()
		else:
			raise Exception('expected object of class CGlottalCycle')
		self.normalized = normalize

	# ---------------------------------------------------------------------- #

	def scaleQuadratic(self, amplitudeScalingPower):
		if amplitudeScalingPower == 1:
			raise Exception("no need to scale -- linear ...")
		if amplitudeScalingPower <= 0:
			raise Exception("invalid parameter")
		valMin = self.data.min()
		valMax = self.data.max()
		for i, y in enumerate(self.data):
			yRel = abs(y)
			limit = valMax
			if y < 0: limit = valMin
			yRel /= float(limit)
			yRel = numpy.power(yRel, amplitudeScalingPower)
			yNew = yRel * limit
			self.data[i] = yNew

	# ---------------------------------------------------------------------- #






		
