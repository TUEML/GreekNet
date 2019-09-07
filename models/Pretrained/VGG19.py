import sys
sys.path.append("../")  # noqa
import keras_helper
import keras_callbacks
from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf

if __name__ == "__main__":

    log_dir, checkpoints = keras_helper.ask_what_to_do_with_logfiles(
        log_dir='logs/vgg19/',
        ckpt="checkpoints/vgg19/{epoch:02d}.ckpt")

    dg = keras_helper.DataGenerators(train_path="../../Dataset/sample_sizes/64_images/",
                                     test_path="../../Dataset/test/",
                                     enable_augmentation=False,
                                     preprocessing_fn=preprocess_input)
    train_generator, test_generator, tensorboard_generator = dg.create_data_generators(224)
    train_samples, test_samples = dg.get_samples()
    batch_size = dg.get_batch_size()

    model = keras_helper.create_model(VGG19(input_shape=(224, 224, 3), include_top=False), train_up_to=2)
    model.compile(Adam(lr=0.0005), loss="categorical_crossentropy", metrics=["accuracy", keras_helper.top_3_acc])

    print(model.summary())

    tensorboard_callback = keras_callbacks.TensorBoardImage(log_dir=log_dir, model=model, DataGenerator=dg)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints, period=20)

    model.fit_generator(train_generator, steps_per_epoch=train_samples // batch_size, epochs=100,
                        validation_data=test_generator, validation_steps=test_samples // batch_size,
                        callbacks=[tensorboard_callback, checkpoint_callback])
