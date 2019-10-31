#!/usr/bin/env python

# ----------------------------------------------------------------
# 3D-Conv-2D-Pool-UNet Testing Indoor Cases
# Written by Haiyang Jiang
# Mar 20th 2019
# ----------------------------------------------------------------

import os, time

import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from skvideo.io import vwrite, vread


from network import network
from config import *


import sys
if len(sys.argv) <= 1:
    test_case = 3
else:
    try:
        test_case = int(sys.argv[1])
    except ValueError:
        test_case = 3


if test_case == 0:
    file_list = FILE_LIST
    directory = 'train_set_results/'
elif test_case == 1:
    file_list = VALID_LIST
    directory = 'validation_set_results/'
elif test_case == 2:
    file_list = TEST_LIST
    directory = 'test_set_results/'
else:
    file_list = CUSOMIZED_LIST
    directory = 'customized_test_results/'


TEST_RESULT_DIR = RESULT_DIR + directory
FILE_LIST = file_list


with open(FILE_LIST) as f:
    text = f.readlines()

train_ids = [line.strip().split(' ')[0] for line in text]
in_paths = [line.strip().split(' ')[2] for line in text]
gt_paths = [line.strip().split(' ')[1] for line in text]


def equalize_histogram(image, number_bins=256):
    image_histogram, bins = np.histogram(image.flatten(), number_bins)
    cdf = image_histogram.cumsum()
    cdf = (number_bins - 1) * cdf / cdf[-1] # normalize
    
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    
    return image_equalized.reshape(image.shape)


def process_video(sess, in_image, out_image, in_file, raw, out_file=None):
    input_patch = raw

    if DEBUG:
        print '[DEBUG] (begining of preocess_video) input_patch.shape:', input_patch.shape

    i = 0
    j = 0
    k = 0
    step = 1 - OVERLAP
    output = np.zeros([input_patch.shape[0], input_patch.shape[1] * 2, input_patch.shape[2] * 2, 3], dtype='uint16')
    i_range, j_range, k_range = input_patch.shape[0:3]
    weights = np.zeros(output.shape, dtype='uint8')
    
    # 16 bit
    max_val = 65535.0
    scaling_factor = max_val
    val_type = 'uint16'
    
    input_patch = equalize_histogram(input_patch, int(max_val) + 1)
    
    
    done = False
    while i < i_range:
        if i + TEST_CROP_FRAME > i_range:
            if done:
                break
            i = i_range - TEST_CROP_FRAME
            done = True
        print '[INFO] processing frame', i
        j = 0
        while j < j_range:
            k = 0
            while k < k_range:
                temp = input_patch[i: i + TEST_CROP_FRAME, j: j + TEST_CROP_HEIGHT, k: k + TEST_CROP_WIDTH, :]
                network_input = np.float32(np.expand_dims(temp, axis=0))
                network_input = np.minimum(network_input / scaling_factor, 1.0)
                if DEBUG:
                    print '[DEBUG] network_input.shape:', network_input.shape
                network_output = sess.run(out_image, feed_dict={in_image: network_input})
                if DEBUG:
                    print '[DEBUG] network_output.shape:', network_output.shape

                if i + TEST_CROP_FRAME > i_range:
                    temp = network_output[0, :i_range - i, :, :, :]
                else:
                    temp = network_output[0, :, :, :, :]
                network_output = np.minimum(np.maximum(temp, 0), 1)
                output[i: i + TEST_CROP_FRAME, j * 2: (j + TEST_CROP_HEIGHT) * 2, k * 2: (k + TEST_CROP_WIDTH) * 2, :] += (network_output * OUT_MAX).astype('uint16')
                weights[i: i + TEST_CROP_FRAME, j * 2: (j + TEST_CROP_HEIGHT) * 2, k * 2: (k + TEST_CROP_WIDTH) * 2, :] += 1
                k += int(TEST_CROP_WIDTH * step)
            j += int(TEST_CROP_HEIGHT * step)
        i += int(TEST_CROP_FRAME * step)

    output = (output / weights).astype('uint8')

    if out_file is None:
        out_file = os.path.basename(in_file)[:-4] + '.mp4'
        if DEBUG:
            print '[DEBUG] out_file:', out_file
    print '[PROCESS] Processing done. Saving...',
    t0 = time.time()
    vwrite(TEST_RESULT_DIR + out_file, output)
    t1 = time.time()
    print 'done. ({:.3f}s)'.format(t1 - t0)


def main():
    sess = tf.Session()
    in_image = tf.placeholder(tf.float32, [None, TEST_CROP_FRAME, None, None, 4])
    gt_image = tf.placeholder(tf.float32, [None, TEST_CROP_FRAME, None, None, 3])
    out_image = network(in_image)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    if not os.path.isdir(TEST_RESULT_DIR):
        os.makedirs(TEST_RESULT_DIR)

    for i, file0 in enumerate(in_paths):
        t0 = time.time()
        # raw = vread(file0)
        raw = np.load(file0)
        if raw.shape[0] > MAX_FRAME:
            print 'Video with shape', raw.shape, 'is too large. Splitted.'
            count = 0
            begin_frame = 0
            while begin_frame < raw.shape[0]:
                t1 = time.time()
                print 'processing segment %d ...' % (count + 1),
                new_filename = '.'.join(file0.split('.')[:-1] + [str(count)] + file0.split('.')[-1::])
                process_video(sess, in_image, out_image, new_filename, raw[begin_frame: begin_frame + MAX_FRAME, :, :, :])
                count += 1
                begin_frame += MAX_FRAME
                print '\t{}s'.format(time.time() - t1)
        else:
            process_video(sess, in_image, out_image, file0, raw, out_file=train_ids[i] + '.mp4')
        print train_ids[i], '\t{}s'.format(time.time() - t0)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print 'total time: {}s'.format(time.time() - t0)
