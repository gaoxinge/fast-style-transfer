import os
import scipy.misc
import numpy as np
import tensorflow as tf


def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)


def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = _get_img(style_path, img_size=new_shape)
    return style_target


def get_img(src, img_size=False):
    img = scipy.misc.imread(src, mode='RGB')
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    if img_size:
        img = scipy.misc.imresize(img, img_size)
    return img


def get_img2(filename):
    img_raw = tf.io.read_file(filename)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
    return img_tensor


def parse_fn(filename):
    img_raw = tf.io.read_file(filename)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
    img_tensor = tf.image.resize(img_tensor, [256, 256])
    img_tensor = tf.cast(img_tensor, tf.float32)
    return img_tensor


def exists(p, msg):
    assert os.path.exists(p), msg


def list_files(in_path):
    files = []
    for dirpath, dirnames, filenames in os.walk(in_path):
        files.extend(filenames)
        break
    return files
