import matplotlib.pyplot as plt
import pandas as pd
import shutil

def save_plot_history(history, save_path,split_num=0):
    
    historyInDrivePath = save_path + 'split_'+ str(split_num) +'_history.csv'
    pd.DataFrame(history).to_csv(historyInDrivePath) #gdrive
    pd.DataFrame(history).to_csv('split_'+str(split_num)+'_history.csv')  #local

    print('plotting and saving train test graphs...')
    #print(history.keys())
    # summarize history for accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.savefig('split_'+str(split_num)+'_accuracy.png',bbox_inches='tight') #local
    plt.savefig(save_path+'accuracy.png',bbox_inches='tight') #gdrive
    

    
    # summarize history for loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.savefig('split_'+str(split_num)+'_loss.png',bbox_inches='tight')  #local
    plt.savefig(save_path+'loss.png',bbox_inches='tight')  #gdrive



def remove_data(dataset):
    print('removing the data folders...')
    shutil.rmtree('/{}/videos'.format(dataset))
    shutil.rmtree('/{}/processed'.format(dataset))
    print('work on split number',split_num,'done!')