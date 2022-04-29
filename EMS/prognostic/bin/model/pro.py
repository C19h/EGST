# author:c19h
# datetime:2022/4/26 14:20
import numpy as np
import pandas as pd
import scipy.stats


class Prognostic:
    def __init__(self, data, region, window=10, amplitude=0.01, time_step=30, threshold=0.95, tolerance=5):
        self.data = data
        self.region = region
        self.correction_window = window
        self.correction_amplitude = amplitude
        self.time_step = time_step
        self.threshold = threshold
        self.tolerance = tolerance
        self.__correct_func = self.__generate_correct()

    def __generate_correct(self):
        cor = np.append(self.correction_amplitude * np.ones(int(self.correction_window / 2)),
                        -self.correction_amplitude * np.ones(int(self.correction_window / 2)))
        correct = np.tile(cor, int(self.time_step / self.correction_window))
        return correct

    def __cal_person(self, y1, y2):
        person_list_cor = []
        for i in range(len(y1) - self.time_step + 1):
            tem_cor = scipy.stats.pearsonr(y1[i:i + self.time_step] + self.__correct_func,
                                           y2[i:i + self.time_step] + self.__correct_func)[0]
            person_list_cor.append(tem_cor)
        return person_list_cor

    def correlation_analysis_mean(self, data):
        assmble_cor = []
        mean = data.mean(axis=0)
        for i in range(len(data)):
            person_cor = self.__cal_person(data.iloc[i], mean)
            assmble_cor.append(np.array(person_cor))
        return assmble_cor

    def correlation_analysis(self, data):
        assmble_cor = []
        for i in range(len(data)):
            if i == len(data) - 1:
                person_cor = self.__cal_person(data.iloc[i], data.iloc[0])
            else:
                person_cor = self.__cal_person(data.iloc[i], data.iloc[i + 1])
            assmble_cor.append(np.array(person_cor))
        return assmble_cor

    def inner(self):
        res = self.correlation_analysis_mean(self.data)
        num_abnormal = []
        for r in res:
            index_tem = np.count_nonzero(r < self.threshold)
            num_abnormal.append(index_tem)
        num_abnormal = np.array(num_abnormal)
        index = np.where(num_abnormal > self.tolerance)
        return np.array(res), num_abnormal, index

    def execute(self):
        res, ab, index = self.inner()
        return res, ab, index
