import sys
sys.path.append("../")  # noqa
import keras_helper
import keras_callbacks
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf

if __name__ == "__main__":

    log_dir, checkpoints = keras_helper.ask_what_to_do_with_logfiles(
        log_dir='./logs/resnet_final/',
        ckpt="checkpoints/resnet_final/{epoch:02d}.ckpt")

    dg = keras_helper.DataGenerators(train_path="../../Dataset/sample_sizes/64_images/",
                                     test_path="../../Dataset/test/",
                                     enable_augmentation=True,
                                     preprocessing_fn=preprocess_input)
    train_generator, test_generator, tensorboard_generator = dg.create_data_generators(224, 15, 0.15, 0.15, 0.25)
    train_samples, test_samples = dg.get_samples()
    batch_size = dg.get_batch_size()

    # model = keras_helper.create_model(ResNet50(input_shape=(224, 224, 3), include_top=False), train_up_to=6)
    model = load_model("checkpoints/resnet_final/200.ckpt", compile=False)
    keras_helper.set_untrainable(model, 10)
    model.compile(Adam(lr=0.000125), loss="categorical_crossentropy", metrics=["accuracy", keras_helper.top_3_acc])

    print(model.summary())

    tensorboard_callback = keras_callbacks.TensorBoardImage(log_dir=log_dir, model=model, DataGenerator=dg)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints, period=20)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(keras_callbacks.lr_scheduler, verbose=1)

    model.fit_generator(train_generator, steps_per_epoch=train_samples // batch_size, epochs=300,
                        validation_data=test_generator, validation_steps=test_samples // batch_size,
                        callbacks=[tensorboard_callback, checkpoint_callback, lr_callback], initial_epoch=200)
