import pandas as pd
import numpy as np
import math as math
import json
import matplotlib.pyplot as plt
import string
import random
import warnings

def genCleanLinearSegment(startTime, endTime, lowVal, highVal, walkPath='direct', revertPer=1.0, walkWidth=0.03):
    """Top-level function to generate a linear segment of points"""
    seqLen = endTime - startTime + 1
    slope = (highVal - lowVal) / (endTime - startTime)
    if(walkPath == 'direct'):
        return genDirectLinearSegment(lowVal, seqLen, slope)
    elif(walkPath == 'wander-slope-rev'):
        return genSlopeRevertingSegment(lowVal, seqLen, slope, revertPer, walkWidth)
    else:
        raise ValueError("walkPath needs to be one of {direct, wander-slope-rev}")
        
def genDirectLinearSegment(lowVal, seqLen, slope):
    """Primary function to generate a _direct_ linear segment of points"""
    # Generate clean values along slope
    # Rounded to ints (since these are supposed to be frequencies)
    # seqLen = endTime - startTime + 1
    # slope = (highVal - lowVal) / (endTime - startTime)
    rawVals = np.array(range(0, seqLen))
    rawVals = rawVals.astype('float32')
    tval = lowVal
    for t in range(0, seqLen):
        rawVals[t] = np.around(tval)
        tval += slope
    return rawVals.astype('int')

def genSlopeRevertingSegment(lowVal, seqLen, slope, revertPer=1.0, walkWidth=0.03):
    """Primary function to generate wandering linear segment of points - with reversion to the direct slope"""
    # Generate clean values along slope
    # Rounded to ints (since these are supposed to be frequencies)
    #seqLen = endTime - startTime + 1
    #slope = (highVal - lowVal) / (endTime - startTime)
    rawVals = np.array(range(0, seqLen))
    rawVals = rawVals.astype('float32')
    tval = lowVal
    directVal = lowVal
    currSlopeDiffPer = 0
    for t in range(0, seqLen):
        rawVals[t] = np.around(tval)
        # directVal is if we stayed on the pure/clean slope
        directVal += slope
        # tval is where we actually are
        # for the moment, bump it by the chosen slope - then we'll adjust it
        tval += slope
        currSlopeDiffPer = np.absolute(tval - directVal) / directVal
        # print("directVal= ", directVal, "currSlopeDiffPer= ", currSlopeDiffPer)
        # Now, take a random walk with reversion to the slope
        # Note that the currSlopeDiffPer * revertPer can be > 1, so our
        #   probability distribution flattens
        # TODO - Maybe update how this adjust works to a more continuous, asymptotic curve
        if(np.random.uniform(0,1) <= (currSlopeDiffPer * revertPer)):
            # move tval direction closer to directVal - walkWidth
            incr = np.random.uniform(0, (walkWidth*tval)) * np.sign(directVal - tval)
            # print("\t-->Moving towards slope by = ", incr)
        else:
            # move tval in either direction +/- walkWidth
            incr = np.random.uniform((-walkWidth*tval), (walkWidth*tval))
            # print("Moving randomly = ", incr)
        tval += incr
    return rawVals.astype('int')

def addNoiseAndOutliers(intSeq, distro, noiseLvl = 0.1, outlierPer=0, outlierMult=2):
    """Function to add Noise and Outliers - in place - to a segment of values"""
    if(outlierPer < 0):
        raise ValueError("Negative outlierPer (", outlierPer, "). Must be >= zero.")
    if(distro == "normal"):
        for i in range(0, intSeq.size):
            # If outlier, noise is the current value * outlierMult
            #  otherwise, add any noise
            tempOutPe = np.random.uniform(0,1)
            if(tempOutPe <= (0.5 * outlierPer)):
                # outlier has negative impact
                intSeq[i] = np.around(intSeq[i] - (intSeq[i] * (outlierMult-1)))
            elif (tempOutPe <= (outlierPer)):
                # outlier has positive impact
                intSeq[i] = np.around(intSeq[i] + (intSeq[i] * (outlierMult-1)))
            else:
                normStdev = noiseLvl * intSeq[i]
                noise = np.random.normal(0, normStdev)
                intSeq[i] = np.around(intSeq[i] + noise)
    elif(distro == "uniform"):
        for i in range(0, intSeq.size):
            # If outlier, multiply by outlierMult, and then skip noise
            #  otherwise, add any noise
            tempOutPe = np.random.uniform(0,1)
            if(tempOutPe <= (0.5 * outlierPer)):
                # outlier has negative impact
                intSeq[i] = np.around(intSeq[i] - (intSeq[i] * (outlierMult-1)))
            elif (tempOutPe <= (outlierPer)):
                # outlier has positive impact
                intSeq[i] = np.around(intSeq[i] + (intSeq[i] * (outlierMult-1)))
            else:
                uniLow = -0.5 * noiseLvl * intSeq[i]
                uniHigh = 0.5 * noiseLvl * intSeq[i]
                noise = np.random.uniform(uniLow, uniHigh)
                intSeq[i] = np.around(intSeq[i] + noise)
    else:
        raise ValueError("distro needs to be one of {normal, uniform}")
    return intSeq

class timeSegment:
    def __init__(self, iStart=1, iEnd=100, iLow=250, iHigh=250, iShape='flat', iWalkPath = 'direct', iRevertPer=3.0, 
                 iWalkWidth=0.03, iNoiseD = 'uniform', iNoiseLvl=0.1, iOutlierP=0.03, iOutlierM=1.1, iDisconF=False):
        if((type(iStart) != int) or (type(iEnd) != int) or (type(iLow) != int) or (type(iHigh) != int)):
            raise TypeError("All of {iStart, iEnd, iLow, iHigh} must be specified as int")
        if(iStart > iEnd):
            raise ValueError("iStart should be lte iEnd")
        self.startTime = iStart
        self.endTime = iEnd
        self.lowVal = iLow
        self.highVal = iHigh
        if iShape not in ['flat', 'posSlope', 'negSlope']:
            raise ValueError("iShape should be one of {flat, posSlope, negSlope}")
        if(iShape == 'flat' and (iLow != iHigh)):
            raise ValueError("When Shape=='flat' iLow should be eq (==) iHigh")
        elif(iShape == 'posSlope' and (iLow > iHigh)):
            raise ValueError("When Shape=='posSlope' iLow should be lte (< =) iHigh")
        elif(iShape == 'negSlope' and (iLow < iHigh)):
            raise ValueError("When Shape=='negSlope' iLow should be gte (> =) iHigh")
        self.shape = iShape
        if (iWalkPath not in ['direct', 'wander-slope-rev']):
            raise ValueError("iWalkPath must be one of {direct, wander-slope-rev}")
        self.walkPath = iWalkPath
        self.revertPer = iRevertPer
        self.walkWidth = iWalkWidth
        self.noiseDistro = iNoiseD
        self.noiseLvl = iNoiseLvl
        self.outlierPercent = iOutlierP
        self.outlierMult = iOutlierM
        self.disconFlag = iDisconF

    def __str__(self):
        print(self.startTime, self.endTime, self.lowVal, self.highVal, self.shape, self.walkPath, end=" ")
        return(str(self.disconFlag))
    
    def print_full(self):
        print("startTime=", self.startTime)
        print("endTime=", self.endTime)
        print("lowVal=", self.lowVal)
        print("highVal=", self.highVal)
        print("shape=", self.shape)
        print("Path:\n  walkPath=", self.walkPath)
        print("  revertPer=", self.revertPer)
        print("  walkWidth=", self.walkWidth)
        print("Noise:\n  noiseDistro=", self.noiseDistro)
        print("  noiseLvl=", self.noiseLvl)
        print("Outlier:\n  outlierPercent=", self.outlierPercent)
        print("  outlierMult=", self.outlierMult)
        print("disconFlag=", self.disconFlag)
        
def buildSeriesFromSeq(tSequences):
    """Top-Level function to convert a list of segments into a combined time series"""
    if not tSequences:
        raise ValueError("buildSeriesFromSeq takes a list of at least one segment")
        
    # For first pass through loop, ensure we don't trigger the overlapping segment error check
    prevEndTime = tSequences[0].startTime - 1
    
    # For first pass through loop, ensure we don't trigger discontinuity error check
    prevHighVal = tSequences[0].lowVal
    prevWalkPath = 'direct'
    
    timeSeries = np.empty(0, int)
    
    for iseg in tSequences:
        # Check that this segment doesn't overlap the previous one
        if (iseg.startTime <= prevEndTime):
            raise IndexError("Segment startTime=", iseg.startTime, "starts at/before the end of the previous segment")
        
        # Check that this segment is correctly tagged if discontinuous from the previous segment
        # If disconFlag is True, we expect a jump between the lowVal of this segment and the highVal of the previous segment
        if ((iseg.lowVal == prevHighVal) and iseg.disconFlag):
            raise ValueError("Dicontinuity error. Values match but disconFlag is True. Previous highVal=", prevHighVal, "Current lowVal=", iseg.lowVal)
        
        # If disconFlag is False, we expect the lowVal of this segment to be equal to the highVal of the previous segment
        elif((iseg.lowVal != prevHighVal) and (not iseg.disconFlag)):
            if(prevWalkPath != 'wander-slope-rev'):
                raise ValueError("Dicontinuity error. Values do not match and disconFlag is False. Previous highVal=", 
                                 prevHighVal, "Current lowVal=", iseg.lowVal)
 
            #unless... the previous segment was a Wandering segment.  In which case, its highVal is not precisely determined
            else: 
                warnings.warn("Potential dicontinuity error. Values do not match and disconFlag is False, BUT previous segment walkPath == wander-slope-rev")
        
        thisSeg = genCleanLinearSegment(iseg.startTime, iseg.endTime, iseg.lowVal, iseg.highVal, iseg.walkPath, iseg.revertPer, iseg.walkWidth)
        # print(thisSeg)
        thisSeg = addNoiseAndOutliers(thisSeg, iseg.noiseDistro, noiseLvl = iseg.noiseLvl, outlierPer=iseg.outlierPercent, outlierMult=iseg.outlierMult)
        # print(thisSeg)
        timeSeries = np.concatenate((timeSeries, thisSeg))
        
        # Update stats for "previous" segment
        prevEndTime = iseg.endTime
        prevHighVal = iseg.highVal
        prevWalkPath = iseg.walkPath
    return timeSeries
