import tensorflow as tf
import io
import cv2
import numpy as np
from PIL import Image


class TensorBoardImage(tf.keras.callbacks.TensorBoard):

    def __init__(self, log_dir, model, DataGenerator):
        self.log_dir = log_dir
        self.mapping_dict = DataGenerator.get_label_mapping_dict()
        self.model = model
        self.preprocess_input = DataGenerator.preprocessing_fn
        self.tensorboard_generator = DataGenerator.tensorboard_generator
        self.size = DataGenerator.get_image_shape()[0]
        super().__init__(log_dir=log_dir)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs={}):
        imgs, labels = self.predict_for_tensorboard()
        summary_list_train = []
        summary_list_val = []
        for i, img in enumerate(imgs):
            img2 = Image.fromarray(img.astype('uint8'))
            output = io.BytesIO()
            img2.save(output, format='PNG')

            image = tf.Summary.Image(height=self.size, width=self.size, encoded_image_string=output.getvalue())
            summary_list_val.append(tf.Summary.Value(tag=f'Prediction/img{i}', image=image))

        summary_list_val.append(tf.Summary.Value(tag='Accuracy', simple_value=logs['val_acc']))
        summary_list_train.append(tf.Summary.Value(tag='Accuracy', simple_value=logs['acc']))

        summary_list_val.append(tf.Summary.Value(tag='Top-3 Accuracy', simple_value=logs['val_top_3_acc']))
        summary_list_train.append(tf.Summary.Value(tag='Top-3 Accuracy', simple_value=logs['top_3_acc']))

        writer1 = tf.summary.FileWriter(logdir=self.log_dir+"/Training", flush_secs=5)
        writer2 = tf.summary.FileWriter(logdir=self.log_dir+"/Validation", flush_secs=5)
        writer1.add_summary(tf.Summary(value=summary_list_train), global_step=epoch)
        writer1.flush()
        writer2.add_summary(tf.Summary(value=summary_list_val), global_step=epoch)
        writer2.flush()

    def predict_for_tensorboard(self):
        imgs, labels = next(self.tensorboard_generator)
        to_preprocess = imgs.copy()

        pred_imgs = self.preprocess_input(to_preprocess)
        for c, i in enumerate(pred_imgs):
            pred = np.argsort(self.model.predict(i.reshape(1, self.size, self.size, 3)))[0][-3:][::-1]
            label = self.mapping_dict[labels[c].argmax()]

            for cc, j in enumerate(pred):
                imgs[c] = cv2.putText(imgs[c], self.mapping_dict[j], (10, 30 + 20 * cc),
                                      cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, [0, 0, 0], 2)

            imgs[c] = cv2.putText(imgs[c], label, (120, 20),
                                  cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, [0, 0, 0], 2)
        return imgs, labels


def lr_scheduler(epoch, lr):

    if epoch % 50 == 0:
        lr /= 2

    return lr
