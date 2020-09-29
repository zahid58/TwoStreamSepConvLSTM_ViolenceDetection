from skimage import segmentation, measure
import numpy as np
import random
import numbers
import scipy
import PIL
import cv2
from PIL import ImageOps
import math
# -----------------------------------------

# Geometric Transformations


class GaussianBlur(object):
    """
    Augmenter to blur images using gaussian kernels.
    Args:
        sigma (float): Standard deviation of the gaussian kernel.
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, clip):

        if isinstance(clip[0], np.ndarray):
            return [scipy.ndimage.gaussian_filter(img, sigma=self.sigma, order=0) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.filter(PIL.ImageFilter.GaussianBlur(radius=self.sigma)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))


class ElasticTransformation(object):
    """
    Augmenter to transform images by moving pixels locally around using
    displacement fields.
    See
        Simard, Steinkraus and Platt
        Best Practices for Convolutional Neural Networks applied to Visual
        Document Analysis
        in Proc. of the International Conference on Document Analysis and
        Recognition, 2003
    for a detailed explanation.
    Args:
        alpha (float): Strength of the distortion field. Higher values mean
        more "movement" of pixels.
        sigma (float): Standard deviation of the gaussian kernel used to
        smooth the distortion fields.
        order (int): Interpolation order to use. Same meaning as in
        `scipy.ndimage.map_coordinates` and may take any integer value in
        the range 0 to 5, where orders close to 0 are faster.
        cval (int): The constant intensity value used to fill in new pixels.
        This value is only used if `mode` is set to "constant".
        For standard uint8 images (value range 0-255), this value may also
        come from the range 0-255. It may be a float value, even for
        integer image dtypes.
        mode : Parameter that defines the handling of newly created pixels.
        May take the same values as in `scipy.ndimage.map_coordinates`,
        i.e. "constant", "nearest", "reflect" or "wrap".
    """
    def __init__(self, alpha=0, sigma=0, order=3, cval=0, mode="constant",
                 name=None, deterministic=False):
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.cval = cval
        self.mode = mode

    def __call__(self, clip):

        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        result = []
        nb_images = len(clip)
        for i in range(nb_images):
            image = clip[i]
            image_first_channel = np.squeeze(image[..., 0])
            indices_x, indices_y = self._generate_indices(image_first_channel.shape, alpha=self.alpha, sigma=self.sigma)
            result.append(self._map_coordinates(
                clip[i],
                indices_x,
                indices_y,
                order=self.order,
                cval=self.cval,
                mode=self.mode))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in result]
        else:
            return result

    def _generate_indices(self, shape, alpha, sigma):
        assert (len(shape) == 2),"shape: Should be of size 2!"
        dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    def _map_coordinates(self, image, indices_x, indices_y, order=1, cval=0, mode="constant"):
        assert (len(image.shape) == 3),"image.shape: Should be of size 3!"
        result = np.copy(image)
        height, width = image.shape[0:2]
        for c in range(image.shape[2]):
            remapped_flat = scipy.ndimage.interpolation.map_coordinates(
                image[..., c],
                (indices_x, indices_y),
                order=order,
                cval=cval,
                mode=mode
            )
            remapped = remapped_flat.reshape((height, width))
            result[..., c] = remapped
        return result



class PiecewiseAffineTransform(object):
    """
    Augmenter that places a regular grid of points on an image and randomly
    moves the neighbourhood of these point around via affine transformations.
     Args:
         displacement (init): gives distorted image depending on the valuse of displacement_magnification and displacement_kernel
         displacement_kernel (init): gives the blury effect
         displacement_magnification (float): it magnify the image
    """
    def __init__(self, displacement=0, displacement_kernel=0, displacement_magnification=0):
        self.displacement = displacement
        self.displacement_kernel = displacement_kernel
        self.displacement_magnification = displacement_magnification

    def __call__(self, clip):

        ret_img_group = clip
        if isinstance(clip[0], np.ndarray):
            im_size = clip[0].shape
            image_w, image_h = im_size[1], im_size[0]
        elif isinstance(clip[0], PIL.Image.Image):
            im_size = clip[0].size
            image_w, image_h = im_size[0], im_size[1]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        displacement_map = np.random.rand(image_h, image_w, 2) * 2 * self.displacement - self.displacement
        displacement_map = cv2.GaussianBlur(displacement_map, None,
                                            self.displacement_kernel)
        displacement_map *= self.displacement_magnification * self.displacement_kernel
        displacement_map = np.floor(displacement_map).astype('int32')

        displacement_map_rows = displacement_map[..., 0] + np.tile(np.arange(image_h), (image_w, 1)).T.astype('int32')
        displacement_map_rows = np.clip(displacement_map_rows, 0, image_h - 1)

        displacement_map_cols = displacement_map[..., 1] + np.tile(np.arange(image_w), (image_h, 1)).astype('int32')
        displacement_map_cols = np.clip(displacement_map_cols, 0, image_w - 1)

        if isinstance(clip[0], np.ndarray):
            return [img[(displacement_map_rows.flatten(), displacement_map_cols.flatten())].reshape(img.shape) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [PIL.Image.fromarray(np.asarray(img)[(displacement_map_rows.flatten(), displacement_map_cols.flatten())].reshape(np.asarray(img).shape)) for img in clip]



class Superpixel(object):
    """
    Completely or partially transform images to their superpixel representation.
    Args:
        p_replace (int) : Defines the probability of any superpixel area being
        replaced by the superpixel.
        n_segments (int): Target number of superpixels to generate.
        Lower numbers are faster.
        interpolation (str): Interpolation to use. Can be one of 'nearest',
        'bilinear' defaults to nearest
    """

    def __init__(self, p_replace=0, n_segments=0, max_size=360,
                 interpolation="bilinear"):
        self.p_replace = p_replace
        self.n_segments = n_segments
        self.interpolation = interpolation


    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]
        # TODO this results in an error when n_segments is 0
        replace_samples = np.tile(np.array([self.p_replace]), self.n_segments)
        avg_image = np.mean(clip, axis=0)
        segments = segmentation.slic(avg_image, n_segments=self.n_segments,
                                     compactness=10)
        if not np.max(replace_samples) == 0:
            clip = [self._apply_segmentation(img, replace_samples, segments) for img in clip]
        if is_PIL:
            return [PIL.Image.fromarray(img) for img in clip]
        else:
            return clip

    def _apply_segmentation(self, image, replace_samples, segments):
        nb_channels = image.shape[2]
        image_sp = np.copy(image)
        for c in range(nb_channels):
            # segments+1 here because otherwise regionprops always misses
            # the last label
            regions = measure.regionprops(segments + 1,
                                          intensity_image=image[..., c])
            for ridx, region in enumerate(regions):
                # with mod here, because slic can sometimes create more 
                # superpixel than requested. replace_samples then does 
                # not have enough values, so we just start over with the
                # first one again.
                if replace_samples[ridx % len(replace_samples)] == 1:
                    mean_intensity = region.mean_intensity
                    image_sp_c = image_sp[..., c]
                    image_sp_c[segments == ridx] = mean_intensity

        return image_sp


class DynamicCrop(object):
    """
    Crops the spatial area of a video containing most movemnets
    """
    def __init__(self):
        pass

    def normalize(self,pdf):
        mn = np.min(pdf)
        mx = np.max(pdf)
        pdf = (pdf - mn)/(mx - mn)
        sm = np.sum(pdf)
        return pdf/sm

    def __call__(self, video, opt_flows):
        
        if not isinstance(video , np.ndarray):
            video = np.array(video, dtype=np.float32)
            opt_flows = np.array(opt_flows,dtype=np.float32)

        magnitude = np.sum(opt_flows, axis=0)
        magnitude = np.sum(magnitude, axis=-1)
        thresh = np.mean(magnitude)
        magnitude[magnitude < thresh] = 0
        # calculate center of gravity of magnitude map and adding 0.001 to avoid empty value
        x_pdf = np.sum(magnitude, axis=1) + 0.001
        y_pdf = np.sum(magnitude, axis=0) + 0.001
        # normalize PDF of x and y so that the sum of probs = 1
        x_pdf = x_pdf[112:208]
        y_pdf = y_pdf[112:208]
        x_pdf = self.normalize(x_pdf)
        y_pdf = self.normalize(y_pdf)
        # randomly choose some candidates for x and y
        x_points = np.random.choice(a=np.arange(
            112, 208), size=5, replace=True, p=x_pdf)
        y_points = np.random.choice(a=np.arange(
            112, 208), size=5, replace=True, p=y_pdf)
        # get the mean of x and y coordinates for better robustness
        x = int(np.mean(x_points))
        y = int(np.mean(y_points))
        video = video[:, x-112:x+112, y-112:y+112, :]
        opt_flows = opt_flows[:, x-112:x+112, y-112:y+112, :] 
        # get cropped video
        return video , opt_flows 
                 

# _____________________________________


class Add(object):
    """
    Add a value to all pixel intesities in an video.
    Args:
        value (int): The value to be added to pixel intesities.
    """

    def __init__(self, value=0):
        if value > 255 or value < -255:
            raise TypeError('The video is blacked or whitened out since ' +
                            'value > 255 or value < -255.')
        self.value = value

    def __call__(self, clip):

        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            image = clip[i].astype(np.int32)
            image += self.value
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            image = image.astype(np.uint8)
            data_final.append(image.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Multiply(object):
    """
    Multiply all pixel intensities with given value.
    This augmenter can be used to make images lighter or darker.
    Args:
        value (float): The value with which to multiply the pixel intensities
        of video.
    """

    def __init__(self, value=1.0):
        if value < 0.0:
            raise TypeError('The video is blacked out since for value < 0.0')
        self.value = value

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            image = clip[i].astype(np.float64)
            image *= self.value
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            image = image.astype(np.uint8)
            data_final.append(image.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Pepper(object):
    """
    Augmenter that sets a certain fraction of pixel intensities to 0, hence
    they become black.
    Args:
        ratio (int): Determines number of black pixels on each frame of video.
        Smaller the ratio, higher the number of black pixels.
    """
    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            img = clip[i].astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 0, img)
            data_final.append(img.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final

class Salt(object):
    """
    Augmenter that sets a certain fraction of pixel intesities to 255, hence
    they become white.
    Args:
        ratio (int): Determines number of white pixels on each frame of video.
        Smaller the ratio, higher the number of white pixels.
   """
    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            img = clip[i].astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 255, img)
            data_final.append(img.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


# -------------------------------------------

# Temporal Transformations

class TemporalBeginCrop(object):
    """
    Temporally crop the given frame indices at a beginning.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        out = clip[:self.size]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class TemporalCenterCrop(object):
    """
    Temporally crop the given frame indices at a center.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        center_index = len(clip) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(clip))

        out = clip[begin_index:end_index]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class TemporalRandomCrop(object):
    """
    Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        rand_end = max(0, len(clip) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(clip))

        out = clip[begin_index:end_index]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class InverseOrder(object):
    """
    Inverts the order of clip frames.
    """
    def __call__(self, clip):
        nb_images = len(clip)
        return [clip[img] for img in reversed(range(0, nb_images))]


class Downsample(object):
    """
    Temporally downsample a video by deleting some of its frames.
    Args:
        ratio (float): Downsampling ratio in [0.0 <= ratio <= 1.0].
    """
    def __init__(self , ratio=1.0):
        if ratio < 0.0 or ratio > 1.0:
            raise TypeError('ratio should be in [0.0 <= ratio <= 1.0]. ' +
                            'Please use upsampling for ratio > 1.0')
        self.ratio = ratio

    def __call__(self, clip):
        nb_return_frame = int(np.floor(self.ratio * len(clip)))
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=nb_return_frame)]

        return [clip[i-1] for i in return_ind]


class Upsample(object):
    """
    Temporally upsampling a video by deleting some of its frames.
    Args:
        ratio (float): Upsampling ratio in [1.0 < ratio < infinity].
    """
    def __init__(self , ratio=1.0):
        if ratio < 1.0:
            raise TypeError('ratio should be 1.0 < ratio. ' +
                            'Please use downsampling for ratio <= 1.0')
        self.ratio = ratio

    def __call__(self, clip):
        nb_return_frame = int(np.floor(self.ratio * len(clip)))
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=nb_return_frame)]

        return [clip[i-1] for i in return_ind]


class TemporalFit(object):
    """
    Temporally fits a video to a given frame size by
    downsampling or upsampling.
    Args:
        size (int): Frame size to fit the video.
    """
    def __init__(self, size):
        if size < 0:
            raise TypeError('size should be positive')
        self.size = size

    def __call__(self, clip):
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=self.size)]

        return [clip[i-1] for i in return_ind]


class TemporalElasticTransformation(object):
    """
    Stretches or schrinks a video at the beginning, end or middle parts.
    In normal operation, augmenter stretches the beggining and end, schrinks
    the center.
    In inverse operation, augmenter shrinks the beggining and end, stretches
    the center.
    """

    def __call__(self, clip):
        nb_images = len(clip)
        new_indices = self._get_distorted_indices(nb_images)
        return [clip[i] for i in new_indices]

    def _get_distorted_indices(self, nb_images):
        inverse = random.randint(0, 1)

        if inverse:
            scale = random.random()
            scale *= 0.21
            scale += 0.6
        else:
            scale = random.random()
            scale *= 0.6
            scale += 0.8

        frames_per_clip = nb_images

        indices = np.linspace(-scale, scale, frames_per_clip).tolist()
        if inverse:
            values = [math.atanh(x) for x in indices]
        else:
            values = [math.tanh(x) for x in indices]

        values = [x / values[-1] for x in values]
        values = [int(round(((x + 1) / 2) * (frames_per_clip - 1), 0)) for x in values]
        return values