import tensorflow as tf
import config as c
import numpy as np
import json
import os
from model.ResNet import ResNet
from model.ResNet_v2 import ResNet_v2
from utils.data_utils import load_image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_path = 'result/weight/ResNet_50_v2.h5'
image_path = 'ILSVRC2012_val_00000321.JPEG'

if __name__ == '__main__':
    # gpu config
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # get model
    # model = ResNet(50)
    model = ResNet_v2(50)

    # show
    model.build(input_shape=(None,) + c.input_shape)
    model.load_weights(model_path)

    img, _ = load_image(tf.constant(image_path), 0)
    prediction = model(np.array([img]), training=False)
    print(np.shape(prediction))
    label = np.argmax(prediction)

    with open('data/label_to_content.json', 'r') as f:
        label_to_content = f.readline()
        label_to_content = json.loads(label_to_content)

        print('-' * 40)
        print('image: {}\nclassification result:{}\nconfidence:{:.4f}'.format(image_path,
                                                                              label_to_content[str(label)],
                                                                              prediction[0, label]))
        print('-' * 40)