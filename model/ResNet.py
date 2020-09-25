import config as c
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAvgPool2D, BatchNormalization, Dense


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=(1, 1), **kwargs):
        self.strides = strides
        if self.strides != (1, 1):
            self.shortcut_projection = Conv2D(filters, (1, 1), name='projection', padding='same', use_bias=False)
            self.shortcut_bn = BatchNormalization(name='shortcut_bn', momentum=0.9, epsilon=1e-5)

        self.conv_0 = Conv2D(filters, (3, 3), name='conv_0', strides=self.strides, padding='same', use_bias=False)
        self.conv_1 = Conv2D(filters, (3, 3), name='conv_1', padding='same', use_bias=False)
        self.bn_0 = BatchNormalization(name='bn_0', momentum=0.9, epsilon=1e-5)
        self.bn_1 = BatchNormalization(name='bn_1', momentum=0.9, epsilon=1e-5)

        super(BasicBlock, self).__init__(**kwargs)

    def call(self, inputs, training):
        net = self.conv_0(inputs)
        net = self.bn_0(net, training=training)
        net = tf.nn.relu(net)

        net = self.conv_1(net)
        net = self.bn_1(net, training=training)

        if self.strides != (1, 1):
            shortcut = tf.nn.avg_pool2d(inputs, ksize=(2, 2), strides=(2, 2), padding='SAME')
            shortcut = self.shortcut_projection(shortcut)
            shortcut = self.shortcut_bn(shortcut)
        else:
            shortcut = inputs

        net = net + shortcut
        net = tf.nn.relu(net)
        return net

class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=(1, 1), projection=False, **kwargs):
        self.strides = strides
        self.projection = projection
        if self.strides != (1, 1) or self.projection:
            self.shortcut_projection = Conv2D(filters * 4, (1, 1), name='projection', padding='same', use_bias=False)
            self.shortcut_bn = BatchNormalization(name='shortcut_bn', momentum=0.9, epsilon=1e-5)

        self.conv_0 = Conv2D(filters, (1, 1), name='conv_0', padding='same', use_bias=False)
        self.conv_1 = Conv2D(filters, (3, 3), name='conv_1', strides=strides, padding='same', use_bias=False)
        self.conv_2 = Conv2D(filters * 4, (1, 1), name='conv_2', padding='same', use_bias=False)
        self.bn_0 = BatchNormalization(name='bn_0', momentum=0.9, epsilon=1e-5)
        self.bn_1 = BatchNormalization(name='bn_1', momentum=0.9, epsilon=1e-5)
        self.bn_2 = BatchNormalization(name='bn_2', momentum=0.9, epsilon=1e-5)

        super(BottleneckBlock, self).__init__(**kwargs)

    def call(self, inputs, training):
        net = self.conv_0(inputs)
        net = self.bn_0(net, training=training)
        net = tf.nn.relu(net)

        net = self.conv_1(net)
        net = self.bn_1(net, training=training)
        net = tf.nn.relu(net)

        net = self.conv_2(net)
        net = self.bn_2(net, training=training)

        if self.projection:
            shortcut = self.shortcut_projection(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        elif self.strides != (1, 1):
            shortcut = tf.nn.avg_pool2d(inputs, ksize=(2, 2), strides=(2, 2), padding='SAME')
            shortcut = self.shortcut_projection(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs

        net = net + shortcut
        net = tf.nn.relu(net)
        return net


class ResNet(tf.keras.models.Model):
    def __init__(self, layer_num, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        if c.block_type[layer_num] == 'basic block':
            self.block = BasicBlock
        else:
            self.block = BottleneckBlock

        self.conv0 = Conv2D(64, (7, 7), strides=(2, 2), name='conv0', padding='same', use_bias=False)
        self.bn = BatchNormalization(name='bn', momentum=0.9, epsilon=1e-5)

        self.block_collector = []
        for layer_index, (b, f) in enumerate(zip(c.block_num[layer_num], c.filter_num), start=1):
            if layer_index == 1:
                if c.block_type[layer_num] == 'basic block':
                    self.block_collector.append(self.block(f, name='conv1_0'))
                else:
                    self.block_collector.append(self.block(f, projection=True, name='conv1_0'))
            else:
                self.block_collector.append(self.block(f, strides=(2, 2), name='conv{}_0'.format(layer_index)))

            for block_index in range(1, b):
                self.block_collector.append(self.block(f, name='conv{}_{}'.format(layer_index, block_index)))

        self.global_average_pooling = GlobalAvgPool2D()
        self.fc = Dense(c.category_num, name='fully_connected', activation='softmax', use_bias=False)

    def call(self, inputs, training):
        net = self.conv0(inputs)
        net = self.bn(net, training)
        net = tf.nn.relu(net)
        print('input', inputs.shape)
        print('conv0', net.shape)
        net = tf.nn.max_pool2d(net, ksize=(3, 3), strides=(2, 2), padding='SAME')
        print('max-pooling', net.shape)

        for block in self.block_collector:
            net = block(net, training)
            print(block.name, net.shape)

        net = self.global_average_pooling(net)
        print('global average-pooling', net.shape)
        net = self.fc(net)
        print('fully connected', net.shape)
        return net

if __name__ == '__main__':
    model = ResNet(18)
    model.build((None,) + c.input_shape)
    model.summary()

    model.save_weights('ResNet_18.h5', save_format='h5')
