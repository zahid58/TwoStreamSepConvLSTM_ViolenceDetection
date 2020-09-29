import matplotlib.pyplot as plt
import pandas as pd
import shutil
import numpy as np 
import pickle
from tensorflow.keras.callbacks import Callback as CB


class SaveTrainingCurves(CB):

    def __init__(self, save_path = None, split_num = 0, **kargs):
        super(SaveTrainingCurves,self).__init__(**kargs)

        self.save_path = save_path
        self.split_num = split_num        
        historyInDrivePath = self.save_path + 'split_'+ str(self.split_num) + '_history.csv'

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
        historyInDrivePath = self.save_path + 'split_'+ str(self.split_num) + '_history.csv'
        pd.DataFrame(history).to_csv(historyInDrivePath) # gdrive
        pd.DataFrame(history).to_csv('split_'+ str(self.split_num) + '_history.csv')  # local
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
        plt.savefig('split_'+str(self.split_num)+'_accuracy.png',bbox_inches='tight') # local
        plt.savefig( self.save_path + 'split_'+ str(self.split_num) + '_accuracy.png',bbox_inches='tight') # gdrive
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
        plt.savefig('split_'+str(self.split_num)+'_loss.png',bbox_inches='tight')  # local
        plt.savefig( self.save_path + 'split_'+ str(self.split_num) + '_loss.png',bbox_inches='tight')  # gdrive
        plt.close()


def lr_scheduler(epoch, lr):
    decay_rate = 0.5
    decay_step = 5
    if epoch % decay_step == 0 and epoch and lr>6e-05:
        return lr * decay_rate
    return lr

def save_as_csv(data, save_path, filename, split_num = 0):
    print('saving',filename,'in csv format...')
    DrivePath = save_path + 'split_'+ str(split_num) + filename
    pd.DataFrame(data).to_csv(DrivePath) #gdrive
    pd.DataFrame(data).to_csv('split_'+ str(split_num) + filename)  #local 


def save_plot_history(history, save_path,split_num=0,pickle_only=True):
    
    # pickle
    print('saving history in pickle format...')
    historyFile = save_path + 'split_' + str(split_num) + '_history.pickle'
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
    historyInDrivePath = save_path + 'split_'+ str(split_num) + '_history.csv'
    pd.DataFrame(history).to_csv(historyInDrivePath) #gdrive
    pd.DataFrame(history).to_csv('split_'+ str(split_num) + '_history.csv')  #local
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
    plt.savefig('split_'+str(split_num)+'_accuracy.png',bbox_inches='tight') #local
    plt.savefig( save_path + 'split_'+ str(split_num) + '_accuracy.png',bbox_inches='tight') #gdrive
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
    plt.savefig('split_'+str(split_num)+'_loss.png',bbox_inches='tight')  #local
    plt.savefig( save_path + 'split_'+ str(split_num) + '_loss.png',bbox_inches='tight')  #gdrive
    plt.close()

def evaluate_accuracy_method1(file_):
    test_acc = []
    for history in file_:
        test_acc_fold = history['val_acc']
        test_acc_fold = np.array(test_acc_fold,dtype=np.float32)
        test_acc.append(test_acc_fold)
    test_acc =  np.array(test_acc)
    mean_acc =  np.mean(test_acc,axis=0)
    print('--------------------------------------------')
    print('accuracy_evaluation_method_1 : ')
    epoch_index = np.argmax(mean_acc)
    max_mean_acc = np.max(mean_acc)
    print( 'max mean accuracy :',max_mean_acc, '_epoch :', (epoch_index + 1) )
    start_index = max( epoch_index-10 , 0 )
    end_index = min( epoch_index+10 , np.size(mean_acc)-1 )
    hundred_acc = test_acc[:,start_index:end_index]
    print('final accuracy :',np.mean(hundred_acc),'±',np.std(hundred_acc))
    print('--------------------------------------------')



def evaluate_accuracy_method2(file_):
    test_acc = []
    for history in file_:
        test_acc_fold = history['val_acc']
        test_acc_fold = np.array(test_acc_fold,dtype=np.float32)
        test_acc.append(test_acc_fold)
    test_acc = np.array(test_acc)
    print('--------------------------------------------')
    print("accuracy_evaluation_method_2 : (Sudhakaran's method)")
    max_test_acc_fold = np.max(test_acc,axis=1)
    print('max accuracy per fold : ',max_test_acc_fold)
    print('final accuracy :',np.mean(max_test_acc_fold),'±',np.std(max_test_acc_fold))
    print('--------------------------------------------')

