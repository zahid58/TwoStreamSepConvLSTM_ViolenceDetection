import numpy as np
from utils import evaluate_accuracy_method1, evaluate_accuracy_method2
import pickle

dataset='hockey'
savePath = '/gdrive/My Drive/THESIS/Data/results/' + str(dataset)+'/'

history_on_each_split=[]

for i in range(5):
    split_num = i + 1
    filePath = savePath + 'split_' + str(split_num) + '_history.pickle'
    file_ = open(filePath, 'rb')
    history = pickle.load(file_)
    history_on_each_split.append(history)

evaluate_accuracy_method1(file_=history_on_each_split)
evaluate_accuracy_method2(file_=history_on_each_split)