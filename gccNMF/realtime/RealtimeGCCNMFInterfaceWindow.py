from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider
from kivy.uix.button import Button
import logging
import numpy as np
from kivy.clock import Clock
from kivy.garden.graph import Graph
from kivy.garden.graph import MeshLinePlot


class RealtimeGCCNMFInterfaceWindow(GridLayout):
    def __init__(self, params,
                 gccPHATHistory, tdoaHistory, inputSpectrogramHistory, outputSpectrogramHistory, coefficientMaskHistories,
                 togglePlayAudioProcessQueue, togglePlayAudioProcessAck,
                 togglePlayGCCNMFProcessQueue, togglePlayGCCNMFProcessAck,
                 tdoaParamsGCCNMFProcessQueue, tdoaParamsGCCNMFProcessAck):

        super(RealtimeGCCNMFInterfaceWindow, self).__init__()
        self.numTDOAs = params.numTDOAs
        self.tdoaIndexes = np.arange(self.numTDOAs)
        self.targetTDOAIndex = self.numTDOAs / 2.0
        self.tdoaHistory = tdoaHistory
        self.gccPHATHistory = gccPHATHistory
        self.inputSpectrogramHistory = inputSpectrogramHistory
        self.outputSpectrogramHistory = outputSpectrogramHistory
        self.coefficientMaskHistories = coefficientMaskHistories

        self.togglePlayAudioProcessQueue = togglePlayAudioProcessQueue
        self.togglePlayAudioProcessAck = togglePlayAudioProcessAck
        self.togglePlayGCCNMFProcessQueue = togglePlayGCCNMFProcessQueue
        self.togglePlayGCCNMFProcessAck = togglePlayGCCNMFProcessAck
        self.tdoaParamsGCCNMFProcessQueue =tdoaParamsGCCNMFProcessQueue
        self.tdoaParamsGCCNMFProcessAck = tdoaParamsGCCNMFProcessAck
        # self.toggleSeparationGCCNMFProcessQueue = toggleSeparationGCCNMFProcessQueue
        # self.toggleSeparationGCCNMFProcessAck = toggleSeparationGCCNMFProcessAck

        self.timer = Clock.schedule_interval(self.updateSlider, 0.01)
        self.tdoas = np.arange(self.numTDOAs).astype(np.float32)
        self.tdoaT = np.arange(self.numTDOAs)
        self.initWindow()

    def initWindow(self):
        self.rows = 4
        self.targetModeTDOASlider = Slider(orientation='horizontal', min=0, max=100, value=50)
        self.targetModeWindowTDOASlider = Slider(orientation='horizontal', min=0, max=100, value=50)
        self.targetModeWindowTDOASlider.bind(value=self.OnSliderValueChange)

        self.playIconString = 'Play'
        self.pauseIconString = 'Pause'
        self.playButton = Button(text='Play')
        self.playButton.bind(on_press=self.togglePlay)
        self.initGraph()
        self.add_widget(self.targetModeWindowTDOASlider)
        self.add_widget(self.playButton)

    def initGraph(self):
        self.graph = Graph(xlabel='TDOAs', ylabel='Y', x_ticks_minor=10,
                        x_ticks_major=10, y_ticks_major=0.5, x_grid_label=False,
                        padding=5,
                        x_grid=False, y_grid=False, xmin=0, xmax=self.numTDOAs, ymin=-0.5, ymax=1.5)
        self.gaussianplot = MeshLinePlot(color=[1, 0, 0, 1])  # 사용자가 선택하는 target TDOA 정규 분포
        self.calculated_plot = MeshLinePlot(color=[0, 1, 0, 1])  # gccPHAT으로 분석된 TDOA plot

        self.graph.add_plot(self.gaussianplot)
        self.add_widget(self.graph)

    def OnSliderValueChange(self, instance, value):
        self.tdoaRegionChanged()

    def queueParams(self, queue, ack, params, label='params'):
        ack.clear()
        logging.debug('GCCNMFInterface: putting %s' % label)
        queue.put(params)
        logging.debug('GCCNMFInterface: put %s' % label)
        ack.wait()
        logging.debug('GCCNMFInterface: ack received')


    def updateTogglePlayParamsAudioProcess(self, playing):
        self.queueParams(self.togglePlayAudioProcessQueue,
                         self.togglePlayAudioProcessAck,
                         {'start' if playing else 'stop': ''},
                         'audioProcessParameters')

    def updateTogglePlayParamsGCCNMFProcess(self):
        self.queueParams(self.togglePlayGCCNMFProcessQueue,
                         self.togglePlayGCCNMFProcessAck,
                         {'numTDOAs': self.numTDOAs,
                          'dictionarySize': 256},
                          'gccNMFProcessTogglePlayParameters')


    def togglePlay(self, value):
        playing = self.playButton.text == self.playIconString
        self.playButton.text = (self.pauseIconString if playing else self.playIconString)
        logging.info('1111GCCNMFInterface: setting playing: %s' % playing)

        if playing:
            self.timer()
            self.tdoaRegionChanged()
            self.updateTogglePlayParamsGCCNMFProcess()
            self.updateTogglePlayParamsAudioProcess(playing)
        else:
            self.updateTogglePlayParamsAudioProcess(playing)
            self.timer.cancel()

    def getTDOA(self):
        tdoa = self.targetModeWindowTDOASlider.value / 100.0 * self.numTDOAs
        return tdoa

    def getBeta(self):
        lnBeta = 50 / 100.0  # 임의 값 self.targetModeWindowBetaSlider.value() / 100.0
        lnBeta *= 10.0
        lnBeta -= 5.0
        beta = np.exp(lnBeta)
        return beta

    def getNoiseFloor(self):
        noiseFloorValue = 0 / 100.0  # 임의 값self.targetModeWindowNoiseFloorSlider.value() / 100.0
        return noiseFloorValue

    def getWindowWidth(self):
        # windowWidth = self.targetModeWindowWidthSlider.value / 100.0 * self.numTDOAs
        windowWidth = 50 / 100.0 * self.numTDOAs # 임의 값
        return windowWidth

    def tdoaRegionChanged(self):
        self.queueParams(self.tdoaParamsGCCNMFProcessQueue,
                         self.tdoaParamsGCCNMFProcessAck,
                         {'targetTDOAIndex': self.getTDOA(),
                          'targetTDOAEpsilon': self.getWindowWidth(),
                          # 'targetTDOAEpsilon': self.targetWindowFunctionPlot.getWindowWidth(),  # targetTDOAEpsilon,
                          'targetTDOABeta': self.getBeta(),
                          'targetTDOANoiseFloor': self.getNoiseFloor()
                          },
                         'gccNMFProcessTDOAParameters (region)')

    def updateSlider(self, value):
        sliderValue = self.tdoaHistory.get()[0] / (self.numTDOAs - 1) * 100
        self.targetModeTDOASlider.value = int(sliderValue)
        self.updatePlot()


    def updatePlot(self):
        self.graph.remove_plot(self.gaussianplot)  # 기존의 plot 삭제, gui로 보기 좋도록
        self.graph.remove_plot(self.calculated_plot)  # 삭제

        mu = self.getTDOA()
        alpha = self.getWindowWidth()
        beta = self.getBeta()
        noiseFloor = self.getNoiseFloor()
        data = generalizedGaussian(self.tdoas, alpha, beta, mu)  # 이것이 현재 들리는 소리의 tdoa를 표현
        data -= min(data)
        data = data / max(data) * (1 - noiseFloor) + noiseFloor

        self.gaussianplot.points = [(x, data[x]) for x in range(0, self.numTDOAs)]
        self.graph.add_plot(self.gaussianplot)

        gccPHATValues = np.squeeze(np.mean(self.gccPHATHistory.values, axis=-1))
        gccPHATValues -= min(gccPHATValues)
        gccPHATValues /= max(gccPHATValues)
        self.calculated_plot.points = [(x, gccPHATValues[x]) for x in range(gccPHATValues.shape[0])]
        self.graph.add_plot(self.calculated_plot)


def generalizedGaussian(x, alpha, beta, mu):
    return np.exp(- (np.abs(x - mu) / alpha) ** beta)


