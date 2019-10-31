#!/usr/bin/env python

# ----------------------------------------------------------------
# 3D-Convolution and 2D-Pooling UNet
# Written by Haiyang Jiang
# Mar 1st 2019
# ----------------------------------------------------------------


import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import DEBUG


# leaky ReLU
def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1[0], deconv_filter, tf.shape(x2[0]), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2[0]], -1)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return tf.expand_dims(deconv_output, axis=0)


# 3D-Conv-2D-Pool UNet
def network(input, depth=3, channel=32, prefix=''):
    depth = min(max(depth, 2), 4)

    conv1 = slim.conv3d(input, channel, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv1_1')
    conv1 = slim.conv3d(conv1, channel, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv1_2')
    pool1 = tf.expand_dims(slim.max_pool2d(conv1[0], [2, 2], padding='SAME'), axis=0)

    conv2 = slim.conv3d(pool1, channel * 2, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv2_1')
    conv2 = slim.conv3d(conv2, channel * 2, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv2_2')
    pool2 = tf.expand_dims(slim.max_pool2d(conv2[0], [2, 2], padding='SAME'), axis=0)

    conv3 = slim.conv3d(pool2, channel * 4, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv3_1')
    conv3 = slim.conv3d(conv3, channel * 4, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv3_2')
    if depth == 2:
        up8 = upsample_and_concat(conv3, conv2, channel * 2, channel * 4)
    else:
        pool3 = tf.expand_dims(slim.max_pool2d(conv3[0], [2, 2], padding='SAME'), axis=0)

        conv4 = slim.conv3d(pool3, channel * 8, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv4_1')
        conv4 = slim.conv3d(conv4, channel * 8, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv4_2')
        if depth == 3:
            up7 = upsample_and_concat(conv4, conv3, channel * 4, channel * 8)
        else:
            pool4 = tf.expand_dims(slim.max_pool2d(conv4[0], [2, 2], padding='SAME'), axis=0)

            conv5 = slim.conv3d(pool4, channel * 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv5_1')
            conv5 = slim.conv3d(conv5, channel * 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv5_2')

            up6 = upsample_and_concat(conv5, conv4, channel * 8, channel * 16)
            conv6 = slim.conv3d(up6, channel * 8, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv6_1')
            conv6 = slim.conv3d(conv6, channel * 8, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv6_2')

            up7 = upsample_and_concat(conv6, conv3, channel * 4, channel * 8)
        conv7 = slim.conv3d(up7, channel * 4, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv7_1')
        conv7 = slim.conv3d(conv7, channel * 4, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv7_2')

        up8 = upsample_and_concat(conv7, conv2, channel * 2, channel * 4)
    conv8 = slim.conv3d(up8, channel * 2, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv8_1')
    conv8 = slim.conv3d(conv8, channel * 2, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, channel, channel * 2)
    conv9 = slim.conv3d(up9, channel, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv9_1')
    conv9 = slim.conv3d(conv9, channel, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv9_2')

    conv10 = slim.conv3d(conv9, 12, [1, 1, 1], rate=1, activation_fn=None, scope=prefix + 'g_conv10')

    out = tf.concat([tf.expand_dims(tf.depth_to_space(conv10[:, i, :, :, :], 2), axis=1) for i in range(conv10.shape[1])], axis=1)
    if DEBUG:
        print '[DEBUG] (network.py) conv10.shape, out.shape:', conv10.shape, out.shape

    return out


# test function for network
def main():
    sess = tf.Session()
    in_image = tf.placeholder(tf.float32, [None, 16, None, None, 4])
    gt_image = tf.placeholder(tf.float32, [None, 16, None, None, 3])
    out_image = []
    out_image += network(in_image),
    out_image += network(in_image, 3, prefix='1'),
    out_image += network(in_image, 2, prefix='2'),
    out_image += network(in_image, 3, 16, prefix='3'),
    from skvideo.io import vread, vwrite
    import numpy as np
    vid = np.load('./0_data/raw/test_data/gt_input/001_00_0001.npy')
    vid = np.expand_dims(np.float32(np.minimum((vid[:16, :256, :256, :] / vid.mean() / 5), 1.0)), axis=0)
    sess.run(tf.global_variables_initializer())
    for i, out in enumerate(out_image):
        print out.shape
        output = sess.run(out, feed_dict={in_image: vid})
        output = (np.minimum(output, 1.0) * 255).astype('uint8')
        print output[0].shape
        vwrite(str(i) + '.mp4', output[0])


if __name__ == '__main__':
    main()
