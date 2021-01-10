import matplotlib.pyplot as plt
import pandas as pd
import shutil
import numpy as np 
import pickle
from tensorflow.keras.callbacks import Callback as CB
import os

class SaveTrainingCurves(CB):

    def __init__(self, save_path = None, **kargs):
        super(SaveTrainingCurves,self).__init__(**kargs)

        self.save_path = save_path   
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)    
        historyInDrivePath = os.path.join(self.save_path , 'history.csv')

        history = None
        try:
            history = pd.read_csv(historyInDrivePath)
            history = history.reset_index().to_dict(orient='list')
        except:
            pass
        if history is not None: 
            self.acc = history['acc']
            self.val_acc = history['val_acc']
            self.loss = history['loss']
            self.val_loss = history['val_loss']
        else:
            self.acc = []
            self.val_acc = []
            self.loss = []
            self.val_loss = []
    
    def on_epoch_end(self, epoch, logs = {}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))  
        history = {'acc':self.acc, 'val_acc':self.val_acc,'loss':self.loss,'val_loss':self.val_loss}
        # csv
        historyInDrivePath = os.path.join(self.save_path ,'history.csv')
        pd.DataFrame(history).to_csv(historyInDrivePath) # gdrive
        pd.DataFrame(history).to_csv('history.csv')  # local
        # graphs
        self.plot_graphs(history)

    def plot_graphs(self, history):
        # accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.grid(True)
        plt.savefig('accuracy.png',bbox_inches='tight') # local
        plt.savefig(os.path.join(self.save_path ,'accuracy.png'),bbox_inches='tight') # gdrive
        plt.close()
        # loss
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.grid(True)
        plt.savefig('loss.png',bbox_inches='tight')  # local
        plt.savefig(os.path.join(self.save_path ,'loss.png'),bbox_inches='tight')  # gdrive
        plt.close()


def lr_scheduler(epoch, lr):
    decay_rate = 0.5
    decay_step = 5
    if epoch % decay_step == 0 and epoch and lr>6e-05:
        print('> setting lr = ',lr * decay_rate)
        return lr * decay_rate
    return lr

def save_as_csv(data, save_path, filename):
    print('saving',filename,'in csv format...')
    DrivePath = save_path + filename
    pd.DataFrame(data).to_csv(DrivePath) #gdrive
    pd.DataFrame(data).to_csv(filename)  #local 


def save_plot_history(history, save_path, pickle_only=True):
    # pickle
    print('saving history in pickle format...')
    historyFile = save_path + 'history.pickle'
    try:
        file_ = open(historyFile, 'wb')
        pickle.dump(history, file_)
        print('saved', historyFile)
    except Exception as e:
        print(e)
    
    if pickle_only:
        return    

    # csv
    print('saving history in csv format...')
    historyInDrivePath = save_path + 'history.csv'
    pd.DataFrame(history).to_csv(historyInDrivePath) #gdrive
    pd.DataFrame(history).to_csv('history.csv')  #local
    print('plotting and saving train test graphs...')
    
    # accuracy graph
    plt.figure(figsize=(10, 6))
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.savefig('accuracy.png',bbox_inches='tight') #local
    plt.savefig( save_path + 'accuracy.png',bbox_inches='tight') #gdrive
    plt.close()

    # loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.savefig('loss.png',bbox_inches='tight')  #local
    plt.savefig( save_path  + 'loss.png',bbox_inches='tight')  #gdrive
    plt.close()

