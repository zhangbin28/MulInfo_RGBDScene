import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from .yolo4.model import preprocess_true_boxes, yolo4_body, yolo4_loss, yolo_eval
from .yolo4.utils import get_random_data, letterbox_image

from PIL import Image

import os

class YoloFinetuneModel:
    def __init__(self,anchors_path, classes_path):
        self.anchors = self.get_anchors(anchors_path)
        self.class_names = self.get_classes(classes_path)
        self.num_classes = len(self.class_names)
        self.num_anchors = len(self.anchors)
        self.input_shape = (416,416)
        self.yolo_weights = 'yolo_based/model_data/yolo4_weight.h5'
        self.score = 0.3
        self.iou = 0.45
        self.model_image_size = (608, 608)
        self.label2object = np.loadtxt(classes_path, dtype=str).tolist()
        pass
    
    def get_classes(self, classes_path):
        '''loads the classes'''
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self, anchors_path):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def train(self, train_path, val_path, out_dir):
        model = self.create_model()

        logging = TensorBoard(log_dir=out_dir)
        checkpoint = ModelCheckpoint(out_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='val_loss', save_weights_only=True, save_best_only=True, period=5)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        with open(train_path) as f:
            train_lines = f.read().splitlines()
        with open(val_path) as f:
            val_lines = f.read().splitlines()

        np.random.seed(10101)
        np.random.shuffle(train_lines)
        np.random.seed(None)
        num_train = len(train_lines)
        num_val = len(val_lines)

        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo4_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(self.data_generator_wrapper(train_lines, batch_size, self.input_shape, self.anchors, self.num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=self.data_generator_wrapper(val_lines, batch_size, self.input_shape, self.anchors, self.num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(out_dir + 'trained_weights_stage_1.h5')

        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 4 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(self.data_generator_wrapper(train_lines, batch_size, self.input_shape, self.anchors, self.num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=self.data_generator_wrapper(val_lines, batch_size, self.input_shape, self.anchors, self.num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(out_dir + 'trained_weights_final.h5')
    
    def test(self, rgb_path, weights_path):
        image = Image.open(rgb_path)
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        sess = K.get_session()

        yolo4_model = yolo4_body(Input(shape=(608, 608, 3)), self.num_anchors//3, self.num_classes)
        yolo4_model.load_weights(weights_path)

        print('{} model, anchors, and classes loaded.'.format(weights_path))

        input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(yolo4_model.output, self.anchors,
                len(self.class_names), input_image_shape,
                score_threshold=self.score)

        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo4_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        
        
        if len(out_scores) == 0:
            scene=None
        else:
            vote = {}
            for j in range(len(out_scores)):
                sc = self.label2object[out_classes[j]].split('@')[1]
                score = out_scores[j]
                if sc in vote:
                    vote[sc] += score
                else:
                    vote[sc] = score
            scene = max(vote, key=vote.get)
        return scene
    
    def test_list(self, rgb_list, weights_path):
        sess = K.get_session()

        yolo4_model = yolo4_body(Input(shape=(608, 608, 3)), self.num_anchors//3, self.num_classes)
        yolo4_model.load_weights(weights_path)

        print('{} model, anchors, and classes loaded.'.format(weights_path))

        input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(yolo4_model.output, self.anchors,
                len(self.class_names), input_image_shape,
                score_threshold=self.score)

        results = []
        for rgb in rgb_list:
            image = Image.open(rgb)
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
            image_data = np.array(boxed_image, dtype='float32')

            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo4_model.input: image_data,
                    input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
            
            
            if len(out_scores) == 0:
                scene=None
            else:
                vote = {}
                for j in range(len(out_scores)):
                    sc = self.label2object[out_classes[j]].split('@')[1]
                    score = out_scores[j]
                    if sc in vote:
                        vote[sc] += score
                    else:
                        vote[sc] = score
                scene = max(vote, key=vote.get)
            results.append(scene)
        return results

    def data_generator(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(batch_size):
                if i==0:
                    np.random.shuffle(annotation_lines)
                image, box = get_random_data(annotation_lines[i], input_shape, random=True)
                image_data.append(image)
                box_data.append(box)
                i = (i+1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(batch_size)

    def data_generator_wrapper(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n==0 or batch_size<=0: return None
        return self.data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

    def create_model(self):
        '''create the training model'''
        freeze_body=2
        K.clear_session() # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w = self.input_shape
        

        y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
            self.num_anchors//3, self.num_classes+5)) for l in range(3)]

        model_body = yolo4_body(image_input, self.num_anchors//3, self.num_classes)
        print('Create YOLOv4 model with {} anchors and {} classes.'.format(self.num_anchors, self.num_classes))

        # load_pretrained
        model_body.load_weights(self.yolo_weights, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(self.yolo_weights))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (250, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        label_smoothing = 0
        use_focal_obj_loss = False
        use_focal_loss = False
        use_diou_loss = True
        use_softmax_loss = False

        model_loss = Lambda(yolo4_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': self.anchors, 'num_classes': self.num_classes, 'ignore_thresh': 0.5,
            'label_smoothing': label_smoothing, 'use_focal_obj_loss': use_focal_obj_loss, 'use_focal_loss': use_focal_loss, 'use_diou_loss': use_diou_loss,
            'use_softmax_loss': use_softmax_loss})(
            [*model_body.output, *y_true])
        model = Model([model_body.input, *y_true], model_loss)

        return model
