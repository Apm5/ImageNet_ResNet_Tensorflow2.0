import os
import tensorflow as tf
import config as c
import numpy as np
from tqdm import tqdm
from utils.data_utils import test_10_crop_iterator
from utils.eval_utils import cross_entropy_batch, l2_loss
from model.ResNet import ResNet
from model.ResNet_v2 import ResNet_v2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@tf.function
def test_step(model, images, labels):
    prediction = model(images, training=False)
    prediction = tf.reduce_mean(prediction, axis=0)
    ce = cross_entropy_batch([labels], [prediction])
    return ce, prediction

def test(model, log_file):
    data_iterator = test_10_crop_iterator()

    sum_ce = 0
    sum_correct_num = 0

    for i in tqdm(range(c.test_num)):
        images, labels = data_iterator.next()
        ce, prediction = test_step(model, images, labels)

        sum_ce += ce
        if np.argmax(prediction) == np.argmax(labels):
            sum_correct_num += 1
        # print('ce: {:.4f}, accuracy: {:.4f}'.format(ce, sum_correct_num / (i + 1)))

    log_file.write('test: cross entropy loss: {:.4f}, l2 loss: {:.4f}, accuracy: {:.4f}\n'.format(sum_ce / c.test_num,
                                                                                                  l2_loss(model),
                                                                                                  sum_correct_num / c.test_num))

if __name__ == '__main__':
    # gpu config
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # get model
    # model = ResNet(50)
    model = ResNet_v2(50)

    # show
    model.build(input_shape=(None,) + c.input_shape)

    if c.load_weight_file is None:
        print('Please fill in the path of model weight in config.py')
    else:
        model.load_weights(c.load_weight_file)

    # test
    with open('result/log/test_log.txt', 'a') as f:
        test(model, f)
