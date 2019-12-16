import cv2
import numpy as np
import config as c

"""
|------|
|      | height, y
|      |
|------|
 width, x
"""

def random_size(image, target_size=None):
    height, width, _ = np.shape(image)
    if target_size is None:
        # for test
        # target size is fixed
        target_size = np.random.randint(*c.short_side_scale)
    if height < width:
        size_ratio = target_size / height
    else:
        size_ratio = target_size / width
    resize_shape = (int(width * size_ratio), int(height * size_ratio))  # width and height in cv2 are opposite to np.shape()
    return cv2.resize(image, resize_shape)

def random_aspect(image):
    height, width, _ = np.shape(image)
    aspect_ratio = np.random.uniform(*c.aspect_ratio_scale)
    if height < width:
        resize_shape = (int(width * aspect_ratio), height)
    else:
        resize_shape = (width, int(height * aspect_ratio))
    return cv2.resize(image, resize_shape)

def random_crop(image):
    height, width, _ = np.shape(image)
    input_height, input_width, _ = c.input_shape
    crop_x = np.random.randint(0, width - input_width)
    crop_y = np.random.randint(0, height - input_height)
    return image[crop_y: crop_y + input_height, crop_x: crop_x + input_width, :]

def random_flip(image):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    return image

def random_hsv(image):
    random_h = np.random.uniform(*c.hue_delta)
    random_s = np.random.uniform(*c.saturation_scale)
    random_v = np.random.uniform(*c.brightness_scale)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 0] = image_hsv[:, :, 0] + random_h % 360.0  # hue
    image_hsv[:, :, 1] = np.minimum(image_hsv[:, :, 1] * random_s, 1.0)  # saturation
    image_hsv[:, :, 2] = np.minimum(image_hsv[:, :, 2] * random_v, 255.0)  # brightness

    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

def random_pca(image):
    alpha = np.random.normal(0, c.pca_std, size=(3,))
    offset = np.dot(c.eigvec * alpha, c.eigval)
    image = image + offset
    return np.maximum(np.minimum(image, 255.0), 0.0)

def normalize(image):
    for i in range(3):
        image[..., i] = (image[..., i] - c.mean[i]) / c.std[i]
    return image

def center_crop(image):
    height, width, _ = np.shape(image)
    input_height, input_width, _ = c.input_shape
    crop_x = (width - input_width) // 2
    crop_y = (height - input_height) // 2
    return image[crop_y: crop_y + input_height, crop_x: crop_x + input_width, :]

def test_10_crop(image):
    """
    Standard 10 crop for test.
    Crop 4 corner and 1 center.
    Then flip it.
    """
    height, width, _ = np.shape(image)
    input_height, input_width, _ = c.input_shape
    center_crop_x = (width - input_width) // 2
    center_crop_y = (height - input_height) // 2

    images = []
    images.append(image[:input_height, :input_width, :])  # left top
    images.append(image[:input_height, -input_width:, :])  # right top
    images.append(image[-input_height:, :input_width, :])  # left bottom
    images.append(image[-input_height:, -input_width:, :])  # right bottom
    images.append(image[center_crop_y: center_crop_y + input_height, center_crop_x: center_crop_x + input_width, :])

    image = cv2.flip(image, 1)
    images.append(image[:input_height, :input_width, :])  # left top
    images.append(image[:input_height, -input_width:, :])  # right top
    images.append(image[-input_height:, :input_width, :])  # left bottom
    images.append(image[-input_height:, -input_width:, :])  # right bottom
    images.append(image[center_crop_y: center_crop_y + input_height, center_crop_x: center_crop_x + input_width, :])

    return np.array(images, dtype=np.float32)
