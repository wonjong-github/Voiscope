from kivy.app import App
import logging
import ctypes
import numpy as np

from multiprocessing import Event, Queue, Array #다중프로세스간 동기화를 위한 이벤트 객체 / 객체전달 / 공유메모리

from gccNMF.realtime.defs import DEFAULT_AUDIO_FILE, DEFAULT_CONFIG_FILE
from gccNMF.realtime.utils import SharedMemoryCircularBuffer, OverlapAddProcessor
from gccNMF.realtime.config import getGCCNMFConfigParams
from gccNMF.realtime.audioProcessor import PyAudioStreamProcessor as AudioStreamProcessor
from gccNMF.realtime.gccNMFProcessor import GCCNMFProcess
from gccNMF.realtime.RealtimeGCCNMFInterfaceWindow import RealtimeGCCNMFInterfaceWindow


class Voiscope(App):
    def init(self): #프로그램 초기 설정파일
        audioPath = DEFAULT_AUDIO_FILE
        configPath = DEFAULT_CONFIG_FILE
        self.params = getGCCNMFConfigParams(audioPath, configPath)

        self.initQueuesAndEvents()
        self.initSharedArrays(self.params)
        self.initHistoryBuffers(self.params)
        self.initProcesses(self.params)

    def initQueuesAndEvents(self):  #프로세스틀 핸들링하기 위한 변수
        self.togglePlayAudioProcessQueue = Queue()
        self.togglePlayAudioProcessAck = Event()
        self.togglePlayGCCNMFProcessQueue = Queue()
        self.togglePlayGCCNMFProcessAck = Event()
        # self.toggleSeparationGCCNMFProcessQueue = Queue()
        # self.toggleSeparationGCCNMFProcessAck = Event()
        self.tdoaParamsGCCNMFProcessQueue = Queue()
        self.tdoaParamsGCCNMFProcessAck = Event()

        self.processFramesEvent = Event()
        self.processFramesDoneEvent = Event()
        self.terminateEvent = Event()

    def initSharedArrays(self, params):
        self.inputFramesArray = Array(ctypes.c_double, params.numChannels * params.blockSize)  # 공유 메모리
        self.inputFrames = np.frombuffer(self.inputFramesArray.get_obj()).reshape((params.numChannels, -1))
        self.outputFramesArray = Array(ctypes.c_double, params.numChannels * params.blockSize)  # 공유 메모리
        self.outputFrames = np.frombuffer(self.outputFramesArray.get_obj()).reshape((params.numChannels, -1))  # 공유 메모리

    def initHistoryBuffers(self, params):
        self.gccPHATHistory = SharedMemoryCircularBuffer((params.numTDOAs, params.numTDOAHistory))
        self.tdoaHistory = SharedMemoryCircularBuffer((1, params.numTDOAHistory))
        self.inputSpectrogramHistory = SharedMemoryCircularBuffer((params.numFreq, params.numSpectrogramHistory))
        self.outputSpectrogramHistory = SharedMemoryCircularBuffer((params.numFreq, params.numSpectrogramHistory))
        self.coefficientMaskHistories = {}
        for size in params.dictionarySizes:
            self.coefficientMaskHistories[size] = SharedMemoryCircularBuffer((size, params.numSpectrogramHistory))

    def initProcesses(self, params):
        self.audioProcess = AudioStreamProcessor(params.numChannels, params.sampleRate, params.windowSize,
                                                 params.hopSize, params.blockSize, params.deviceIndex,
                                                 self.togglePlayAudioProcessQueue, self.togglePlayAudioProcessAck,
                                                 self.inputFrames, self.outputFrames, self.processFramesEvent,
                                                 self.processFramesDoneEvent, self.terminateEvent,
                                                 self.inputFramesArray, self.outputFramesArray)
        self.oladProcessor = OverlapAddProcessor(params.numChannels, params.windowSize, params.hopSize,
                                                 params.blockSize, params.windowsPerBlock, self.inputFrames,
                                                 self.outputFrames, self.inputFramesArray, self.outputFramesArray)

        self.gccNMFProcess = GCCNMFProcess(self.oladProcessor, params.sampleRate, params.windowSize,
                                           params.windowsPerBlock, params.dictionariesW, params.dictionaryType,
                                           params.dictionarySize, params.numHUpdates,
                                           params.microphoneSeparationInMetres, False,#params.localizationEnabled,
                                           params.localizationWindowSize,
                                           self.gccPHATHistory, self.tdoaHistory, self.inputSpectrogramHistory,
                                           self.outputSpectrogramHistory, self.coefficientMaskHistories,
                                           self.tdoaParamsGCCNMFProcessQueue, self.tdoaParamsGCCNMFProcessAck,
                                           self.togglePlayGCCNMFProcessQueue, self.togglePlayGCCNMFProcessAck,
                                           # self.toggleSeparationGCCNMFProcessQueue,
                                           # self.toggleSeparationGCCNMFProcessAck,
                                           self.processFramesEvent, self.processFramesDoneEvent, self.terminateEvent)

        self.audioProcess.start() #audioProcess.run() 실행
        logging.info("\n\nAudio Process Start\n\n")
        self.gccNMFProcess.start()

    def on_stop(self):
        try:
            logging.info('Window closed')
            self.terminateEvent.set()

            self.audioProcess.join()
            logging.info('Audio process joined')

            self.gccNMFProcess.join()
            logging.info('GCCNMF process joined')
        finally:
            self.audioProcess.terminate()
            self.gccNMFProcess.terminate()

    def build(self):
        self.init()
        return RealtimeGCCNMFInterfaceWindow(self.params, self.gccPHATHistory, self.tdoaHistory, self.inputSpectrogramHistory, self.outputSpectrogramHistory, self.coefficientMaskHistories,
                                                                  self.togglePlayAudioProcessQueue, self.togglePlayAudioProcessAck,
                                                                  self.togglePlayGCCNMFProcessQueue, self.togglePlayGCCNMFProcessAck,
                                                                  self.tdoaParamsGCCNMFProcessQueue, self.tdoaParamsGCCNMFProcessAck)


# if __name__ == '__main__':
#     Voiscope().run()
