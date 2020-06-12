from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import apply_affine_transform
import numpy as np
import os
from time import time
import cv2
import random

class DataGenerator(Sequence):
    """Data Generator inherited from keras.utils.Sequence
    Args: 
        directory: the path of data set, and each sub-folder will be assigned to one class
        batch_size: the number of data points in each batch
        shuffle: whether to shuffle the data per epoch
    Note:
        If you want to load file with other data format, please fix the method of "load_data" as you want
    """
    def __init__(self, directory, batch_size=1, shuffle=False, data_augmentation=True,one_hot=False,target_frames=20):
        # Initialize the params
        self.batch_size = batch_size
        self.directory = directory
        self.shuffle = shuffle
        self.data_aug = data_augmentation
        self.one_hot = one_hot
        self.target_frames = target_frames
        # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        self.X_path, self.Y_dict = self.search_data() 
        # Print basic statistics information
        self.print_stats()
        return None
        
    def search_data(self):
        X_path = []
        Y_dict = {}
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory))
        one_hots = to_categorical(range(len(self.dirs)))
        for i,folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory,folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path,file)
                # append the each file path, and keep its label  
                X_path.append(file_path)
                if self.one_hot:
                    Y_dict[file_path] = one_hots[i]
                else:
                    Y_dict[file_path] = i
        return X_path, Y_dict
    
    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.X_path)
        self.n_classes = len(self.dirs)
        self.indexes = np.arange(len(self.X_path))
        np.random.shuffle(self.indexes)
        # Output states
        print("Found {} files belonging to {} classes.".format(self.n_files,self.n_classes))
        for i,label in enumerate(self.dirs):
            print('%10s : '%(label),i)
        return None
    
    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        """Get the data of each batch
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)
        return batch_x, batch_y

    def on_epoch_end(self):
        # shuffle the data at each end of epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_path):
        # load data into memory, you can change the np.load to any method you want
        batch_x = [self.load_data(x) for x in batch_path]
        batch_y = [self.Y_dict[x] for x in batch_path]
        # transfer the data format and take one-hot coding for labels
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y
      
    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        #mean = [0.485, 0.456, 0.406]
        #std = [0.229, 0.224, 0.225]
        return (data-mean) / std
    
    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video    
    
    def uniform_sampling(self, video, target_frames=20):
        # get total frames of input video and calculate sampling interval 
        len_frames = int(len(video))
        interval = int(np.ceil(len_frames/target_frames))
        # init empty list for sampled video and 
        sampled_video = []
        for i in range(0,len_frames,interval):
            sampled_video.append(video[i])     
        # calculate numer of padded frames and fix it 
        num_pad = target_frames - len(sampled_video)
        padding = []
        if num_pad>0:
            for i in range(-num_pad,0):
                try: 
                    padding.append(video[i])
                except:
                    padding.append(video[0])
            sampled_video += padding     
        # get sampled video
        return np.array(sampled_video, dtype=np.float32)
    
    def random_clip(self, video, target_frames=20):
        start_point = np.random.randint(len(video)-target_frames)
        return video[start_point:start_point+target_frames]
            
    def color_jitter(self,video,prob=0.5):
        # range of s-component: 0-1
        # range of v component: 0-255
        s = np.random.rand()
        if s > prob:
            return video
        s_jitter = np.random.uniform(-0.2,0.2)
        v_jitter = np.random.uniform(-30,30)
        for i in range(len(video)):
            hsv = cv2.cvtColor(video[i], cv2.COLOR_RGB2HSV)
            s = hsv[...,1] + s_jitter
            v = hsv[...,2] + v_jitter
            s[s<0] = 0
            s[s>1] = 1
            v[v<0] = 0
            v[v>255] = 255
            hsv[...,1] = s
            hsv[...,2] = v
            video[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return video


    def crop_corner(self, video ,prob=0.5):
        s = np.random.rand()
        if s > prob:
            return video
        frame_size = np.size(video, axis = 1)
        corner_keys = ["Left_up","Right_down","Right_up","Left_down","Center"]
        corner = random.choice(corner_keys)
        percentages = [.8,.9]
        percentage = random.choice(percentages)
        resize = int(frame_size*percentage)
        if(corner =="Left_up"):
            x_start = 0
            x_end = resize
            y_start = 0
            y_end = resize
        if (corner == "Right_down"):
            x_start = frame_size-resize
            x_end = frame_size
            y_start = frame_size-resize
            y_end = frame_size
        if(corner =="Right_up"):
            x_start = 0
            x_end = resize
            y_start = frame_size-resize
            y_end = frame_size
        if (corner == "Left_down"):
            x_start = frame_size-resize
            x_end = frame_size
            y_start = 0
            y_end = resize
        if (corner == "Center"):
            half = int(frame_size*(1-percentage))
            x_start = half
            x_end = frame_size-half
            y_start = half
            y_end = frame_size-half
        
        for i in range(np.shape(video)[0]):
            x = cv2.resize(video[i,y_start:y_end,x_start:x_end, :], (frame_size, frame_size)).astype(np.float32)
            video[i] = x
        return video


    def random_shear(self, video, intensity, prob = 0.5, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0., interpolation_order=1):
        s = np.random.rand()
        if s > prob:
            return video
        shear = np.random.uniform(-intensity, intensity)
        for i in range(np.shape(video)[0]):
            x = apply_affine_transform(video[i], shear=shear, channel_axis=channel_axis,
                                fill_mode=fill_mode, cval=cval,
                                order=interpolation_order)
            video[i]=x
        return video
    

    def random_shift(self ,video, wrg, hrg, prob =0.5,row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0., interpolation_order=1):
        s = np.random.rand()
        if s > prob:
            return video        
        h, w = video.shape[1], video.shape[2]
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w
        for i in range(np.shape(video)[0]):
            x = apply_affine_transform(video[i], tx=tx, ty=ty, channel_axis=channel_axis,
                                fill_mode=fill_mode, cval=cval,
                                order=interpolation_order)
            video[i] = x
        return video

    def random_rotation(self, video, rg, prob =0.5,row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0., interpolation_order=1):
        s = np.random.rand()
        if s > prob:
            return video        
        theta = np.random.uniform(-rg,rg)
        for i in range(np.shape(video)[0]):
            x = apply_affine_transform(video[i], theta=theta, channel_axis=channel_axis,
                                fill_mode=fill_mode, cval=cval,
                                order=interpolation_order)
            video[i] = x
        return video

    def load_data(self, path):
        # load the processed .npy files
        data = np.load(path, mmap_mode='r')
        data = np.float32(data)
        # sampling 20 frames uniformly from the entire video
        data = self.uniform_sampling(video=data, target_frames=self.target_frames)
        # whether to utilize the data augmentation
        if  self.data_aug:
            data = self.color_jitter(data, prob=1)
            data = self.random_flip(data, prob=0.5)
            data = self.crop_corner(data, prob=0.5)
            data = self.random_shift(data, wrg = .2, hrg= .2, prob = 0.8)
            #data = self.random_shear(data,intensity=10,prob=0.4)
            data = self.random_rotation(data, rg=15, prob=0.7)
        # normalize rgb images 
        data = self.normalize(data)
        return data



### Demo code
if __name__=="__main__":
    dataset = 'hockey'
    train_generator = DataGenerator(directory='../Datasets/{}/train'.format(dataset), 
                                    batch_size=batch_size, 
                                    data_augmentation=True,
                                    shuffle=False,
                                    one_hot = False,
                                    target_frames=20)
    print(train_generator)


    