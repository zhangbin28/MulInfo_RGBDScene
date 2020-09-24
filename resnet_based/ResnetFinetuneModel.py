

import json
import keras
from keras.layers import Dense, Conv1D, Flatten
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD
import keras.backend as K
import numpy as np
import skimage

from resnet import resnet50
from Depth2HHA.toHHA import toHHA

class ResnetFinetuneModel:
    def __init__(self, label2name_path):
        self.label2name = json.load(open(label2name_path))
        self.class_num = len(self.label2name.keys())
        self.resnet_weights = 'resnet_based/model/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        self.feature_model = self.ResnetModel()
    
    def ResnetModel(self):
        resnet_model = resnet50.resnet50_model(self.resnet_weights)
        feature_layer = 'avg_pool'
        features_model = Model(inputs=resnet_model.input,
                            outputs=resnet_model.get_layer(feature_layer).output)
        return features_model

    def FintuneModel(self, input_shape):
        classifier_model = Sequential()
        classifier_model.add(Dense(self.class_num*64, activation='sigmoid', input_shape = input_shape))
        classifier_model.add(Dense(self.class_num, activation='sigmoid'))
        return classifier_model

    def HHACoding(self, depth_path):
        return toHHA(depth_path)

    def Preprocess(self, im):

        """
        Preprocesses image array for classifying using ImageNet trained Resnet-152 model
        :param im: RGB, RGBA float-type image or grayscale image
        :return: ready image for passing to a Resnet model
        """

        # Detect invalid images
        if im is None or not hasattr(im, 'shape') or len(im.shape) < 2: return None

        # If grayscale, convert to RGB
        if len(im.shape) == 2:
            im = np.asarray(np.dstack((im, im, im)), dtype=np.uint8)

        # Remove alpha channel, if necessary
        if im.shape[2] == 4:
            im = im[:, :, 0:3]

        if len(im.shape) < 2:
            print("Wrong image shape", im.shape)

        # RGB to BGR
        im = im[:, :, ::-1]

        # Resize and scale values to <0, 255>
        im = skimage.transform.resize(im, (224, 224), mode='constant').astype(np.float32)
        im *= 255

        # Subtract ImageNet mean
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68

        # Add a dimension
        im = np.expand_dims(im, axis=0)

        return im

    def ResnetFeature(self, rgb_path, depth_path):
        hha = self.HHACoding(depth_path)

        img = self.Preprocess(skimage.io.imread(rgb_path))
        feature_rgb = self.feature_model.predict(img).flatten()

        img = self.Preprocess(skimage.io.imread(depth_path))
        feature_depth = self.feature_model.predict(img).flatten()

        img = self.Preprocess(hha[:,:,0])
        feature_sn = self.feature_model.predict(img).flatten()

        feature = np.append(feature_rgb, feature_depth, axis=0)
        feature = np.append(feature, feature_sn, axis=0)

        feature = np.expand_dims(feature, axis=0)
        return feature

    def ResnetFeatureList(self, path):
        features_rgb = []
        features_depth = []
        features_sn = []
        labels = []

        data = np.loadtxt(path, dtype=str, delimiter='\t')
        for i in range(data.shape[0]):
            hha = self.HHACoding(data[i][1])

            img = self.Preprocess(skimage.io.imread(data[i][0]))
            features_rgb.append(self.feature_model.predict(img).flatten())

            img = self.Preprocess(skimage.io.imread(data[i][1]))
            features_depth.append(self.feature_model.predict(img).flatten())

            img = self.Preprocess(hha[:,:,0])
            features_sn.append(self.feature_model.predict(img).flatten())

            labels.append(data[i][2])
        
        features = np.append(features_rgb, features_depth, axis = 1)
        features = np.append(features, features_sn, axis=1)
        labels = keras.utils.np_utils.to_categorical(labels)

        return features, labels

    def train(self, train_path, val_path, out_weights):
        train_features, train_labels = self.ResnetFeatureList(train_path)
        val_features, val_labels     = self.ResnetFeatureList(val_path)

        lr_decay = ReduceLROnPlateau(factor=0.9, patience=1, verbose=1)
        checkpointer = ModelCheckpoint(filepath=out_weights, save_best_only=True, verbose=1)
        print(type(lr_decay), type(checkpointer))

        opt = SGD(lr=0.1)
        K.clear_session()
        classifier_model = self.FintuneModel(train_features.shape[1:])
        classifier_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        classifier_model.summary()
        classifier_model.fit(train_features, train_labels,
                                epochs=500,
                                batch_size=256,
                                validation_data=(val_features, val_labels),
                                callbacks=[lr_decay, checkpointer],
                                shuffle = True)

    def test(self, rgb_path, depth_path, weights):
        features = self.ResnetFeature(rgb_path, depth_path)

        K.clear_session()
        classifier_model = self.FintuneModel(features.shape[1:])
        classifier_model.load_weights(weights)

        prediction = classifier_model.predict(features)
        result = int(np.argmax(prediction))

        return self.label2name[str(result)]