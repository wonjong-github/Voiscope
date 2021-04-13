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

import logging
import numpy as np
from multiprocessing import Process
from time import sleep
import time as tm

from gccNMF.realtime.wavfile import pcm2float, float2pcm


class PyAudioStreamProcessor(Process):
    def __init__(self, numChannels, sampleRate, windowSize, hopSize, blockSize, deviceIndex,
                 togglePlayQueue, togglePlayAck, inputFrames, outputFrames, processFramesEvent, processFramesDoneEvent, terminateEvent, inputFramesArray, outputFramesArray):
        super(PyAudioStreamProcessor, self).__init__()

        self.numChannels = numChannels
        self.sampleRate = sampleRate
        self.windowSize = windowSize
        self.hopSize = hopSize
        self.blockSize = blockSize
        
        self.togglePlayQueue = togglePlayQueue
        self.togglePlayAck = togglePlayAck
        self.inputFrames = inputFrames
        self.inputFramesArray = inputFramesArray
        self.outputFrames = outputFrames
        self.outputFramesArray = outputFramesArray
        self.processFramesEvent = processFramesEvent
        self.processFramesDoneEvent = processFramesDoneEvent
        self.terminateEvent = terminateEvent
        self.deviceIndex = deviceIndex  # 사용 안함
        
        self.numBlocksPerBuffer = 8

        self.processingTimes = []
        self.underflowCounter = 0
        
        self.fileName = None
        self.audioStream = None
        self.pyaudio = None

        self.tempFrames = []
        
    def run(self):
        #os.nice(-20)
        
        lastPrintTime = tm.time()
        
        while True:
            currentTime = tm.time()            

            # 프로세스 종료
            if self.terminateEvent.is_set():
                logging.debug('AudioStreamProcessor: received terminate')
                if self.audioStream:
                    self.audioStream.close()
                    logging.debug('AudioStreamProcessor: stream stopped')
                return
            
            if not self.togglePlayQueue.empty():
                parameters = self.togglePlayQueue.get()
                logging.debug('AudioStreamProcessor: received togglePlayParams')
        
                if 'stop' in parameters: self.stopStream()
                elif 'start' in parameters: self.startStream()        
                
                logging.debug('AudioStreamProcessor: processed togglePlayParams')
                self.togglePlayAck.set()
                logging.debug('AudioStreamProcessor: ack set')
            elif currentTime - lastPrintTime >= 2:
                if len(self.processingTimes) != 0:
                    logging.info( 'Processing times (min/max/avg): %f, %f, %f' % (np.min(self.processingTimes), np.max(self.processingTimes), np.mean(self.processingTimes)) )
                    lastPrintTime = currentTime
                    del self.processingTimes[:]
            else:
                sleep(0.01)
    
    def filePlayerCallback(self, in_data, numFrames, time_info, status):

        startTime = tm.time()
        inputBuffer = in_data
        inputIntArray = np.frombuffer(inputBuffer, dtype='<i2')  # < 는 little-endian, i2가 signed 2byte integer
        self.inputFramesArray[:] = pcm2float(inputIntArray)  # float형 으로 변환

        # logging.info('AudioStreamProcessor: setting processFramesEvent')
        self.processFramesDoneEvent.clear()
        # logging.info(self.inputFrames)
        self.processFramesEvent.set()
        # logging.info('AudioStreamProcessor: waiting for processFramesDoneEvent')
        self.processFramesDoneEvent.wait()
        # logging.info('AudioStreamProcessor: done waiting for processFramesDoneEvent')
        # outputFramesArray는 잘 저장되어 있음(float 값)
        tempArray = np.array(self.outputFramesArray)
        Max, Min = np.max(tempArray), np.min(tempArray)
        tempArray = tempArray / max(Max, -Min, 1)
        # float2pcm은 -1에서 1사이에 분포한 float 값들을 정수로 바꾸는 것, 근데 outputFramesArray는 정규화 안되있는 것
        # outputIntArray = float2pcm(self.outputFramesArray)

        outputIntArray = float2pcm(tempArray)
        try:
            outputBuffer = np.getbuffer(outputIntArray)
        except:
            # numpy에서 getbuffer 없어짐, 그래서 tobyte()를 사용한다
            outputBuffer = outputIntArray.tobytes()  # little endian 바이트 저장방식??

        self.processingTimes.append(tm.time() - startTime)
        self.tempFrames.append(outputBuffer)
        return outputBuffer, self.paContinue
    
    def active(self):
        if not self.audioStream:
            return False
        else:
            return self.audioStream.is_active()

    def startStream(self):
        if not self.audioStream:
            logging.info('AudioStreamProcessor: creating stream...')
            # logging.info(self.fileName)
            self.reset()
        logging.info('AudioStreamProcessor: starting stream')
        self.audioStream.start_stream()
        
    def stopStream(self):
        if self.audioStream:
            logging.info('AudioStreamProcessor: stopping stream')
            ##### 녹음 파일 생성 #######
            # import wave
            # w = wave.open("../../data/record_temp.wav", 'wb')
            # w.setnchannels(self.numChannels)
            # w.setsampwidth(self.bytesPerFrame)
            # w.setframerate(self.sampleRate)
            # w.writeframes(b''.join(self.tempFrames))
            # w.close()
            self.audioStream.stop_stream()
            
    def reset(self):
        if self.audioStream:
            logging.info('AudioStreamProcessor: aborting stream')
            self.tempFrames = []
            self.audioStream.close()

        # logging.info('AudioStreamProcessor: call createAudioStream()')
        self.createAudioStream()
        
    def togglePlay(self):
        self.stopStream() if self.active() else self.startStream()

    def createAudioStream(self):
        import pyaudio
        
        if self.pyaudio is None:
            self.pyaudio = pyaudio.PyAudio()
        
        self.paContinue = pyaudio.paContinue

        # 마이크로 입력
        self.numChannels = 2
        self.sampleRate = 8000
        self.bytesPerFrame = 2
        self.bytesPerFrameAllChannels = self.bytesPerFrame * self.numChannels
        self.audioStream = self.pyaudio.open(format=8,  # self.pyaudio.get_format_from_width(self.bytesPerFrame) = 8
                                             channels=2,  # 2
                                             rate=self.sampleRate,  # 16000
                                             frames_per_buffer=self.blockSize,  # 512
                                             input=True,  # 입력
                                             output=True,
                                             stream_callback=self.filePlayerCallback)

        '''
        waveFile = wave.open(self.fileName, 'rb')
        self.numFrames = waveFile.getnframes()
        self.samples = waveFile.readframes(self.numFrames)
        self.sampleRate = waveFile.getframerate()
        self.bytesPerFrame = waveFile.getsampwidth()
        self.bytesPerFrameAllChannels = self.bytesPerFrame * self.numChannels
        self.format = self.pyaudio.get_format_from_width(self.bytesPerFrame)
        # print("numFrames : ", self.numFrames)
        # print("sampleRate : ", self.sampleRate)
        # print("bytesPerFrame : ", self.bytesPerFrame)
        # print("bytesPerFrameAllChannels : ", self.bytesPerFrameAllChannels)
        # print("format : ", self.format)
        waveFile.close()
        self.waveFile = wave.open(self.fileName, 'rb')
        self.sampleIndex = 0
        self.audioStream = self.pyaudio.open(format=self.format,
                                             channels=self.numChannels,
                                             rate=self.sampleRate,
                                             frames_per_buffer=self.blockSize,
                                             output=True,
                                             stream_callback=self.filePlayerCallback)

        '''
        '''
        defaultAudioInfo = self.pyaudio.get_default_input_device_info()
        self.sampleRate = int(defaultAudioInfo["defaultSampleRate"])
        self.numChannels = defaultAudioInfo["maxInputChannels"]
        self.bytesPerFrame = 2
        self.format = self.pyaudio.get_format_from_width(self.bytesPerFrame)
        self.chunk = int(self.sampleRate / 10)  # 44100
        '''


