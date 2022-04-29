# author:c19h
# datetime:2022/4/19 11:23
import pandas as pd
import scipy.stats
import numpy as np


class SecurityWarning:
    def __init__(self, window=10, amplitude=0.01, time_step=30):
        self.correction_window = window
        self.correction_amplitude = amplitude
        self.time_step = time_step
        self.__correct_func = self.__generate_correct()

    def __generate_correct(self):
        cor = np.append(self.correction_amplitude * np.ones(int(self.correction_window / 2)),
                        -self.correction_amplitude * np.ones(int(self.correction_window / 2)))
        correct = np.tile(cor, int(self.time_step / self.correction_window))
        return correct

    def __cal_person(self, y1, y2):
        # person_list = []
        person_list_cor = []
        for i in range(len(y1) - self.time_step + 1):
            # tem = scipy.stats.pearsonr(y1[i:i + self.time_step], y2[i:i + self.time_step])[0]
            tem_cor = scipy.stats.pearsonr(y1[i:i + self.time_step] + self.__correct_func,
                                           y2[i:i + self.time_step] + self.__correct_func)[0]
            # person_list.append(tem)
            person_list_cor.append(tem_cor)
        return person_list_cor

    def correlation_analysis_mean(self, data):
        print('analysis correlation...')
        assmble_cor = []
        mean = data.mean(axis=0)
        for i in range(len(data)):
            person_cor = self.__cal_person(data.iloc[i], mean)
            assmble_cor.append(np.array(person_cor))
        return assmble_cor

    def correlation_analysis(self, data):
        print('analysis correlation...')
        assmble_cor = []
        for i in range(len(data)):
            if i == len(data) - 1:
                person_cor = self.__cal_person(data.iloc[i], data.iloc[0])
            else:
                person_cor = self.__cal_person(data.iloc[i], data.iloc[i + 1])
            assmble_cor.append(np.array(person_cor))
        return assmble_cor
