import numpy as np
from utils import evaluate_accuracy_method1, evaluate_accuracy_method2

a = np.arange(10,130,3)
b = np.arange(70,-10,-2)
c = np.arange(140,20,-3)

history_list= []
history = {'val_acc':a}
history_list.append(history)
history = {'val_acc':b}
history_list.append(history)
history = {'val_acc':c}
history_list.append(history)
history = {'val_acc':b}
history_list.append(history)
history = {'val_acc':a}
history_list.append(history)

#---------

evaluate_accuracy_method1(file_=history_list)
evaluate_accuracy_method2(file_=history_list)