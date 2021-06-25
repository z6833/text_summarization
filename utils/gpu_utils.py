# coding=utf-8
import os
import tensorflow as tf


def config_gpus(gpu_memory=6):

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if not gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print('GPU is not avaiable to be used . CPU is processing ...')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        tf.config.experimental.set_memory_growth(gpus[0], enable=True)
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * gpu_memory)]
            )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f'{len(gpus)} Physical Devices, {len(logical_gpus)} Logical Devices')
        except RuntimeError as e:
            print(e)


