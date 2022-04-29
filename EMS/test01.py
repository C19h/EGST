# author:c19h
# datetime:2022/3/10 14:44
import numpy as np


def cal_similarity_day(a, b):
    a = np.array(a)
    b = np.array(b)
    delta1 = np.maximum(np.abs((a + b) / 2 - a), 0.01)
    delta2 = np.maximum(np.abs(np.sqrt(a * b) - a), 0.01)
    r1 = (np.exp(1 / delta1) + np.exp(1 / delta2)) / 2
    delta3 = np.maximum(np.abs(np.diff(b) - np.diff(a)), 0.01)
    r2 = np.exp(1 / delta3)
    r_mean = 0.5 * np.mean(r1) + 0.5 * np.mean(r2)
    return r_mean


def cal_similarity_day1(a, b):
    a = np.array(a) + 10
    b = np.array(b) + 10
    delta1 = np.maximum(np.abs((a + b) / 2 - a), 0.1)
    delta2 = np.maximum(np.abs(np.sqrt(a * b) - a), 0.1)
    r1 = (np.exp(1 / delta1) + np.exp(1 / delta2)) / 2
    delta3 = np.maximum(np.abs(np.diff(b) - np.diff(a)), 0.1)
    r2 = np.exp(1 / delta3)
    r_mean = 0.5 * np.mean(r1) + 0.5 * np.mean(r2)
    return r_mean


cal_similarity_day([1, 2, 3], [3, 4, 5])
