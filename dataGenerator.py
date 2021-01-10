from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import apply_affine_transform, apply_brightness_shift
import tensorflow as tf
import numpy as np
import os
from time import time
import cv2
import random
import scipy
from videoAugmentator import *

class DataGenerator(Sequence):
    """Data Generator inherited from keras.utils.Sequence
    Args: 
        directory: the path of data set, and each sub-folder will be assigned to one class
        batch_size: the number of data points in each batch
        shuffle: whether to shuffle the data per epoch
    Note:
        If you want to load file with other data format, please fix the method of "load_data" as you want
    """

    def __init__(self, directory, batch_size=1, shuffle=False, data_augmentation=True, one_hot=False, target_frames=32, sample=False, normalize_ = True, background_suppress = True, resize=224, frame_diff_interval=1, dataset=None, mode="both"): 
        # Initialize the params
        self.dataset = dataset
        self.batch_size = batch_size
        self.directory = directory
        self.shuffle = shuffle
        self.data_aug = data_augmentation
        self.one_hot = one_hot
        self.target_frames = target_frames
        self.sample = sample
        self.background_suppress = background_suppress
        print("background suppression:", self.background_suppress)
        self.mode = mode  # ["only_frames","only_differences", "both"]
        self.resize = resize
        self.frame_diff_interval = frame_diff_interval
        self.normalize_ = normalize_
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
        for i, folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
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
        print("Found {} files belonging to {} classes.".format(
            self.n_files, self.n_classes))
        for i, label in enumerate(self.dirs):
            print('%10s : ' % (label), i)
        return None

    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        """Get the data of each batch
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index *
                                    self.batch_size:(index+1)*self.batch_size]
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
        # loading X
        batch_data = []
        batch_diff_data = []
        if self.mode == "both":
            for x in batch_path:
                data, diff_data = self.load_data(x)
                batch_data.append(data)
                batch_diff_data.append(diff_data)
            batch_data = np.array(batch_data)
            batch_diff_data = np.array(batch_diff_data)
        elif self.mode == "only_frames":
            for x in batch_path:
                data = self.load_data(x)
                batch_data.append(data)
            batch_data = np.array(batch_data) 
        elif self.mode == "only_differences":
            for x in batch_path:
                diff_data = self.load_data(x)
                batch_diff_data.append(diff_data)
            batch_diff_data = np.array(batch_diff_data) 
        # loading Y
        batch_y = [self.Y_dict[x] for x in batch_path]
        batch_y = np.array(batch_y)
        if self.mode == "both":
            return [batch_data, batch_diff_data], batch_y
        if self.mode == "only_frames":
            return [batch_data], batch_y            
        if self.mode == "only_differences":
            return [batch_diff_data], batch_y

    def normalize(self, data):
        data = (data / 255.0).astype(np.float32)
        mean = np.mean(data)
        std = np.std(data)
        return (data-mean) / std

    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video

    def uniform_sampling(self, video, target_frames=20):
        # get total frames of input video and calculate sampling interval
        len_frames = video.shape[0]
        interval = int(np.ceil(len_frames/target_frames))
        # init empty list for sampled video and
        sampled_video = []
        for i in range(0, len_frames, interval):
            sampled_video.append(video[i])
        # calculate numer of padded frames and fix it
        num_pad = target_frames - len(sampled_video)
        padding = []
        if num_pad > 0:
            for i in range(-num_pad, 0):
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

    def color_jitter(self, video, prob=1):
        # range of s-component: 0-1
        # range of v component: 0-255
        s = np.random.rand()
        if s > prob:
            return video
        s_jitter = np.random.uniform(-0.3, 0.3)  # (-0.2,0.2)
        v_jitter = np.random.uniform(-40, 40)  # (-30,30)
        for i in range(len(video)):
            hsv = cv2.cvtColor(video[i], cv2.COLOR_RGB2HSV)
            s = hsv[..., 1] + s_jitter
            v = hsv[..., 2] + v_jitter
            s[s < 0] = 0
            s[s > 1] = 1
            v[v < 0] = 0
            v[v > 255] = 255
            hsv[..., 1] = s
            hsv[..., 2] = v
            video[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return video

    def crop_center(self, video, x_crop=10, y_crop=30):
        frame_size = np.size(video, axis=1)
        x = frame_size
        y = frame_size
        x_start = x_crop
        x_end = x - x_crop
        y_start = y_crop
        y_end = y-y_crop
        video = video[:, y_start:y_end, x_start:x_end, :]
        return video

    def random_shear(self, video, intensity, prob=0.5, row_axis=0, col_axis=1, channel_axis=2,
                     fill_mode='nearest', cval=0., interpolation_order=1):
        s = np.random.rand()
        if s > prob:
            return video
        shear = np.random.uniform(-intensity, intensity)

        for i in range(video.shape[0]):
            x = apply_affine_transform(video[i, :, :, :], shear=shear, channel_axis=channel_axis,
                                       fill_mode=fill_mode, cval=cval,
                                       order=interpolation_order)
            video[i] = x
        return video

    def random_shift(self, video, wrg, hrg, prob=0.5, row_axis=0, col_axis=1, channel_axis=2,
                     fill_mode='nearest', cval=0., interpolation_order=1):
        s = np.random.rand()
        if s > prob:
            return video
        h, w = video.shape[1], video.shape[2]
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w

        for i in range(video.shape[0]):
            x = apply_affine_transform(video[i, :, :, :], tx=tx, ty=ty, channel_axis=channel_axis,
                                       fill_mode=fill_mode, cval=cval,
                                       order=interpolation_order)
            video[i] = x
        return video

    def random_rotation(self, video, rg, prob=0.5, row_axis=0, col_axis=1, channel_axis=2,
                        fill_mode='nearest', cval=0., interpolation_order=1):
        s = np.random.rand()
        if s > prob:
            return video
        theta = np.random.uniform(-rg, rg)
        for i in range(np.shape(video)[0]):
            x = apply_affine_transform(video[i, :, :, :], theta=theta, channel_axis=channel_axis,
                                       fill_mode=fill_mode, cval=cval,
                                       order=interpolation_order)
            video[i] = x
        return video

    def random_brightness(self, video, brightness_range):
        if len(brightness_range) != 2:
            raise ValueError(
                '`brightness_range should be tuple or list of two floats. '
                'Received: %s' % (brightness_range,))
        u = np.random.uniform(brightness_range[0], brightness_range[1])
        for i in range(np.shape(video)[0]):
            x = apply_brightness_shift(video[i, :, :, :], u)
            video[i] = x
        return video

    def gaussian_blur(self, video, prob=0.5, low=1, high=2):
        s = np.random.rand()
        if s > prob:
            return video
        sigma = np.random.rand()*(high-low) + low    
        return GaussianBlur(sigma = sigma)(video)    

    def elastic_transformation(self, video, prob=0.5,alpha=0):
        s = np.random.rand()
        if s > prob:
            return video
        return ElasticTransformation(alpha=alpha)(video)   

    def piecewise_affine_transform(self, video, prob=0.5,displacement=3, displacement_kernel=3, displacement_magnification=2):
        s = np.random.rand()
        if s > prob:
            return video
        return PiecewiseAffineTransform(displacement=displacement, displacement_kernel=displacement_kernel, displacement_magnification=displacement_magnification)(video)

    def superpixel(self, video, prob=0.5, p_replace=0, n_segments=0):
        s = np.random.rand()
        if s > prob:
            return video
        return Superpixel(p_replace=p_replace,n_segments=n_segments)(video)    

    def resize_frames(self, video):
        video = np.array(video, dtype=np.float32)
        if (video.shape[1]==self.resize and video.shape[2]==self.resize):
            return video
        resized = []
        for i in range(video.shape[0]):
            x = cv2.resize(
                video[i], (self.resize, self.resize)).astype(np.float32)
            resized.append(x)
        return np.array(resized,dtype=np.float32)
    
    def dynamic_crop(self, video, opt_flows):
        return DynamicCrop()(video, opt_flows)
    
    def random_crop(self, video, prob=0.5):
        s = np.random.rand()
        if s > prob:
            return self.resize_frames(video)
        # gives back a randomly cropped 224 X 224 from a video with frames 320 x 320
        if self.dataset == 'rwf2000' or self.dataset == 'surv':
            x = np.random.choice(
                a=np.arange(112, 320-112), replace=True)
            y = np.random.choice(
                a=np.arange(112, 320-112), replace=True)
            video = video[:, x-112:x+112, y-112:y+112, :]
        else:
            x = np.random.choice(
                a=np.arange(80, 224-80), replace=True)
            y = np.random.choice(
                a=np.arange(80, 224-80), replace=True)
            video = video[:, x-80:x+80, y-80:y+80, :]
            video = self.resize_frames(video)
        return video    

    def background_suppression(self, data):
        video = np.array(data, dtype = np.float32)
        avgBack = np.mean(video, axis=0)
        video = np.abs(video - avgBack)
        return video

    def frame_difference(self, video):
        num_frames = len(video)
        k = self.frame_diff_interval
        out = []
        for i in range(num_frames - k):
            out.append(video[i+k] - video[i])
        return np.array(out,dtype=np.float32)
        
    def pepper(self, video, prob = 0.5, ratio = 100):
        s = np.random.rand()
        if s > prob:
            return video
        return Pepper(ratio=ratio)(video)

    def salt(self, video, prob = 0.5, ratio = 100):
        s = np.random.rand()
        if s > prob:
            return video
        return Salt(ratio=ratio)(video)            

    def inverse_order(self, video, prob = 0.5):
        s = np.random.rand()
        if s > prob:
            return video
        return InverseOrder()(video)    

    def downsample(self, video):
        video = Downsample(ratio=0.5)(video)
        return np.concatenate((video, video), axis = 0)

    def upsample(self, video):
        num_frames = len(video)    
        video = Upsample(ratio=2)(video)
        s = np.random.randint(0,1)
        if s:
            return video[:num_frames]
        else:
            return video[num_frames:]

    def upsample_downsample(self, video, prob=0.5):
        s = np.random.rand()
        if s>prob:
            return video
        s = np.random.randint(0,1)
        if s:
            return self.upsample(video)
        else:
            return self.downsample(video)          

    def temporal_elastic_transformation(self, video, prob=0.5):
        s = np.random.rand()
        if s > prob:
            return video    
        return TemporalElasticTransformation()(video)
        
    def load_data(self, path):

        # load the processed .npy files
        data = np.load(path, mmap_mode='r')
        data = np.float32(data)
        # sampling frames uniformly from the entire video
        if self.sample:
            data = self.uniform_sampling(
                video=data, target_frames=self.target_frames)

        if self.mode == "both":
            frames = True
            differences = True
        elif self.mode == "only_frames":
            frames = True
            differences = False
        elif self.mode == "only_differences":
            frames = False
            differences = True

        # data augmentation
        if self.data_aug:
            data = self.random_brightness(data, (0.5, 1.5))
            data = self.color_jitter(data, prob = 1)
            data = self.random_flip(data, prob=0.50)
            data = self.random_crop(data, prob=0.80)
            data = self.random_rotation(data, rg=25, prob=0.8)
            data = self.inverse_order(data,prob=0.15)
            data = self.upsample_downsample(data,prob=0.5)
            data = self.temporal_elastic_transformation(data,prob=0.2)
            data = self.gaussian_blur(data,prob=0.2,low=1,high=2) 
            
            if differences:
                diff_data = self.frame_difference(data)
            if frames and self.background_suppress:
                data = self.background_suppression(data) ####   
                data = self.pepper(data,prob=0.3,ratio=45)
                data = self.salt(data,prob=0.3,ratio=45)
        else:
            if self.dataset == 'rwf2000' or self.dataset == 'surv':
                data = self.crop_center(data, x_crop=(320-224)//2, y_crop=(320-224)//2)  # center cropping only for test generators
            if differences:
                diff_data = self.frame_difference(data)
            if frames and self.background_suppress:
                data = self.background_suppression(data) ####

        if frames:
            data = np.array(data, dtype=np.float32)
            if self.normalize_:     
                data = self.normalize(data)
            assert (data.shape == (self.target_frames,self.resize, self.resize,3)), str(data.shape)
        if differences:
            diff_data = np.array(diff_data, dtype=np.float32)
            if self.normalize_:
                diff_data = self.normalize(diff_data)
            assert (diff_data.shape == (self.target_frames - self.frame_diff_interval, self.resize, self.resize, 3)), str(data.shape)

        if self.mode == "both":
            return data, diff_data
        elif self.mode == "only_frames":
            return data
        elif self.mode == "only_differences":
            return diff_data



# Demo code
if __name__ == "__main__":
    dataset = 'hockey'
    train_generator = DataGenerator(directory='../Datasets/{}/train'.format(dataset),
                                    batch_size=batch_size,
                                    data_augmentation=True,
                                    shuffle=False,
                                    one_hot=False,
                                    target_frames=20)
    print(train_generator)