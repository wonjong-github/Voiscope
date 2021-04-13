'''
The MIT License (MIT)

Copyright (c) 2017 Sean UN Wood

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: Sean UN Wood
'''

import ctypes
import numpy as np
from numpy import prod, frombuffer, concatenate, exp, abs
from multiprocessing import Array, Value

class SharedMemoryCircularBuffer():
    def __init__(self, shape, initValue=0): # shape = (513, 128)
        self.array = Array( ctypes.c_double, int(prod(shape)) )  # prod는 배열 안의 값 곱, 513 * 128
                                                                 # Array는 공유메모리,
        self.shape = shape
        self.values = frombuffer(self.array.get_obj()).reshape(shape)
        self.values[:] = initValue
        
        self.numValues = self.values.shape[-1]
        
        self.index = Value(ctypes.c_int)    
        self.index.value = 0

    def selfupdate(self, newValues, numNewValues, index):
        # self.values[] 0 128, 256 ... 128 * (newValues.shape[0] - 1 -> 63)
        # for i in range(newValues.shape[0]):
        #     for j in range(numNewValues):
        #         self.array[index + 128 * i] = self.values[i][j]
        self.array[:] = np.array(self.values).flatten()

    def getupdatevalue(self):
        self.values[:] = frombuffer(self.array.get_obj()).reshape(self.shape)

    def set(self, newValues, index=None):
        index = self.index.value if index is None else index 
        numNewValues = newValues.shape[-1]
        if index + numNewValues < self.numValues:
            self.values[..., index:index+numNewValues] = newValues  # values 값을 업데이트, shape (64 X 1),
            self.selfupdate(newValues, numNewValues, index)  # 업데이트 빠르게 다시 해야함
            self.index.value = index + numNewValues
            return self.index.value
        else:
            numAtEnd = self.numValues - index
            numAtStart = numNewValues - numAtEnd
        
            self.values[..., index:] = newValues[..., :numAtEnd]
            self.values[..., :numAtStart] = newValues[..., numAtEnd:]
            self.index.value = numAtStart
            # self.selfupdate()
            return self.index.value
    
    def get(self, index=None):
        index = (self.index.value-1) % self.numValues if index is None else (index % self.numValues)
        # logging.info(self.values)  # 여기서 values 가 전부 0
        self.getupdatevalue()
        return self.values[..., index]

    def getUnraveledArray(self):
        # logging.info(self.values)
        return concatenate( [self.values[:, self.index.value:], self.values[:, :self.index.value]], axis=-1 ) 
        
    def size(self):
        return self.values.shape[-1]


#공유메모리가 중첩되는 부분이다. 중간에 위치에 따른 변환 과정이 있다.
class OverlapAddProcessor(object):
    def __init__(self, numChannels, windowSize, hopSize, blockSize, windowsPerBlock, inputFrames, outputFrames, inputFramesArray, outputFramesArray):
        super(OverlapAddProcessor, self).__init__()
        
        self.numChannels = numChannels
        self.windowSize = windowSize
        self.hopSize = hopSize
        self.blockSize = blockSize
        self.windowsPerBlock = windowsPerBlock
        
        # IPC

        self.inputFrames = inputFrames
        self.inputFramesArray = inputFramesArray
        self.outputFrames = outputFrames
        self.outputFramesArray = outputFramesArray
        
        # Buffers        
        self.numBlocksPerBuffer = 8  # 8
        
        self.inputBufferIndex = 0
        self.inputBufferSize = self.blockSize * self.numBlocksPerBuffer
        self.inputBuffer = np.zeros( (self.numChannels, self.inputBufferSize), np.float32 ) 

        self.outputBufferIndex = 0
        self.outputBufferSize = self.blockSize * self.numBlocksPerBuffer
        self.outputBuffer = np.zeros( (self.numChannels, self.outputBufferSize), np.float32 )        
        
        self.windowedSamples = np.zeros( (self.numChannels, self.windowSize, self.windowsPerBlock), np.float32 )
    
    def processFrames(self, processFramesFunction):  # processFramesFunction은 함수
        ## 뭔가 공유 메모리에 값이 안넘어가고 공유 메모리 가리키는 변수의 reference를 넘기려 해도 다른 메모리 위치를 가리키고 있음
        #startTime = time()
        # inputFrames는 float 값이 들어감
        self.inputBuffer[:, :-self.blockSize] = self.inputBuffer[:, self.blockSize:]  # (numchannels, inputBufferSize)
        # logging.info(self.inputFramesArray.get_obj())
        ########
        # self.inputFrames[:] = pcm2float(self.inputFramesArray).reshape(-1, self.numChannels).T  # 이걸 공유 메모리로 갱신해야 한다
        # self.inputFrames[:] = pcm2float(inputIntArray).reshape(-1, self.numChannels).T inputIntArray는 버퍼값으로 읽어온 것
        # inputBuffer에 inputFrames가 들어감
        ########
        # self.inputBuffer[:, -self.blockSize:] = np.array(self.inputFramesArray).reshape(self.numChannels, -1)
        self.inputBuffer[:, -self.blockSize:] = np.array(self.inputFramesArray).reshape(-1, self.numChannels).T
        # inputBuffer의 마지막 block에 inputFrames 값을 넣는다.
        
        self.outputBuffer[:, :-self.blockSize] = self.outputBuffer[:, self.blockSize:]
        self.outputBuffer[:, -self.blockSize:] = 0
        
        windowIndexes = np.arange(self.inputBufferSize - self.windowSize - (self.windowsPerBlock-1)*self.hopSize, self.inputBufferSize-self.windowSize +1, self.hopSize)
        for i, windowIndex in enumerate(windowIndexes):
            self.windowedSamples[..., i] = self.inputBuffer[:, windowIndex:windowIndex+self.windowSize]

        processedFrames = processFramesFunction(self.windowedSamples)
        
        for i, windowIndex in enumerate(windowIndexes):
            self.outputBuffer[:, windowIndex:windowIndex+self.windowSize] += processedFrames[..., i]

        # outputBuffer에는 제대로 값이 나온다
        self.outputFrames[:] = self.outputBuffer[:, -3*self.blockSize:-2*self.blockSize]
        self.outputFramesArray[:] = self.outputFrames.T.flatten()
        '''outputFramesArray에 값을 넣을 때 무언가 잘못됨, 저장 방식? --> 제대로 저장됨(float형)'''

        #totalTime = time() - startTime
        #logging.info('processFrames took %f' % totalTime)