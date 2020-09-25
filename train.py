import os
import tensorflow as tf
import config as c
from tqdm import tqdm
from tensorflow.keras import optimizers
from utils.data_utils import train_iterator
from utils.eval_utils import cross_entropy_batch, correct_num_batch, l2_loss
from model.ResNet import ResNet
from model.ResNet_v2 import ResNet_v2
from test import test
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class CosineDecayWithWarmUP(tf.keras.experimental.CosineDecay):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_up_step=0, name=None):
        self.warm_up_step = warm_up_step
        super(CosineDecayWithWarmUP, self).__init__(initial_learning_rate=initial_learning_rate,
                                                    decay_steps=decay_steps,
                                                    alpha=alpha,
                                                    name=name)

    @tf.function
    def __call__(self, step):
        if step <= self.warm_up_step:
            return step / self.warm_up_step * self.initial_learning_rate
        else:
            return super(CosineDecayWithWarmUP, self).__call__(step - self.warm_up_step)

@tf.function
def train_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)
        ce = cross_entropy_batch(labels, prediction, label_smoothing=c.label_smoothing)
        l2 = l2_loss(model)
        loss = ce + l2
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return ce, prediction

def train(model, data_iterator, optimizer, log_file):

    sum_ce = 0
    sum_correct_num = 0

    for i in tqdm(range(c.iterations_per_epoch)):
        images, labels = data_iterator.next()
        ce, prediction = train_step(model, images, labels, optimizer)
        correct_num = correct_num_batch(labels, prediction)

        sum_ce += ce * c.batch_size
        sum_correct_num += correct_num
        print('ce: {:.4f}, accuracy: {:.4f}, l2 loss: {:.4f}'.format(ce,
                                                                     correct_num / c.batch_size,
                                                                     l2_loss(model)))

    log_file.write('train: cross entropy loss: {:.4f}, l2 loss: {:.4f}, accuracy: {:.4f}\n'.format(sum_ce / c.train_num,
                                                                                                   l2_loss(model),
                                                                                                   sum_correct_num / c.train_num))


if __name__ == '__main__':
    # gpu config
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # load data
    train_data_iterator = train_iterator()

    # get model
    # model = ResNet(50)
    model = ResNet_v2(50)

    # show
    model.build(input_shape=(None,) + c.input_shape)
    model.summary()
    print('initial l2 loss:{:.4f}'.format(l2_loss(model)))

    # load pretrain
    if c.load_weight_file is not None:
        model.load_weights(c.load_weight_file)
        print('pretrain weight l2 loss:{:.4f}'.format(l2_loss(model)))

    # train
    learning_rate_schedules = CosineDecayWithWarmUP(initial_learning_rate=c.initial_learning_rate,
                                                    decay_steps=c.epoch_num * c.iterations_per_epoch - c.warm_iterations,
                                                    alpha=c.minimum_learning_rate,
                                                    warm_up_step=c.warm_iterations)
    optimizer = optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9)
    for epoch_num in range(c.epoch_num):
        with open(c.log_file, 'a') as f:
            f.write('epoch:{}\n'.format(epoch_num))
            train(model, train_data_iterator, optimizer, f)
            test(model, f)

        model.save_weights(c.save_weight_file, save_format='h5')

        # save intermediate results
        if epoch_num % 5 == 4:
            os.system('cp {} {}_epoch_{}.h5'.format(c.save_weight_file, c.save_weight_file.split('.')[0], epoch_num))
