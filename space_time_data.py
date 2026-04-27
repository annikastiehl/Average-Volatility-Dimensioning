import numpy as np
import math 

def space_time_data(timeData, sampleData, labels = None, sampleRate = None):
    #Check if size of time vector and samples matrix fit together
    if labels is None: #um mit Roessler zu vergleichen. Abrage ob none, da bei roessler none ist.
        if timeData.shape[0] != sampleData.shape[1]:
            raise ValueError('Size of time vector and samples matrix do not fit together')
    else:
        if timeData.shape[0] != sampleData.shape[1]:
            raise ValueError('Size of time vector and samples matrix do not fit together')
    
    # if no labels are given, empty labels are produced
    numSensors = numSensor(sampleData)
    if labels is None:
        labels = ["" for _ in range(numSensors)]

    
    if sampleRate is None:
        sampleRate = numSampleRate(timeData)
    
    # Check if size of labels vector and samples matrix fit together
    #numSamples = Dimension von sampleData 
    #<3: maximal ist dim nur 3 glaube ich! (?)
    if 'EEG' not in labels[0]:
        if len(labels) != numSensors:
            raise ValueError('Size of labels vector and samples matrix do not fit together')
    
    derivativeData = _derivativesignal(sampleData.transpose(), timeData)
    data = {
        "TimeSignal": timeData,
        "SampleSignal": sampleData,
        "derivateData": derivativeData,
        "Labels": labels
    }

    return data 
    
def _derivativesignal(signal: np.ndarray, time_signal: np.ndarray):
    """Compute the derivative of a signal with respect to its sampling rate

    Arguments:
        signal {np.ndarray} -- Signal to be differentiated (timepoints, channels).
        time_signal {np.ndarray} -- Time vector of the signal (timepoints, ).

    Returns:
        np.ndarray -- derivative of the signal (timepoints, channels)

    """
    max = len(signal)
    dt = time_signal[1] - time_signal[0]
    derivative_signal = (signal[2:max, :] - signal[0:max - 2, :]) / (2 * dt)
    first = (signal[1, :] - signal[0, :]) / (1 * dt)
    last = (signal[max - 1, :] - signal[max - 2, :]) / (1 * dt)
    derivative_signal = np.vstack((first, derivative_signal, last))

    return derivative_signal.transpose()


#Calculate the number of samples, sensors and Sample Rate
def numSamples(sampleData):
    numSamples = sampleData.shape[1]
    return numSamples

def numSensor(sampleData): 
    numSensors = len(sampleData)#muss die länge (=Anzahl der Zeilen) ausgeben (1dim: := 1 ==len()) oder: sampleData.shape[1]
    return numSensors

def numSampleRate(timeData):
    sampleRate = round(1/timeData[1] - timeData[0])
    return sampleRate

time = np.array([0,1,2,1,1])
samples = np.array([[1, 5, 6,8,8],
                [4, 7, 2,8,8],
                [3, 1, 9,8,8],
                [1,1,1,8,8]])
samples1 = samples.T
labels = ["sensor", "sensor2", "", ""]

# data = space_time_data(time, samples.transpose())

# print('Derivative data:');
# print(data["derivateData"]);