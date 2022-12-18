import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications, models, layers, optimizers, losses, metrics

from dataset import *


def main(epochs=100, batch_size=1,
         model_summary=False,
         learning_rate_initial=0.005, learning_phases=3,
         input_shape=(256, 256), n_class=2,
         do_shuffle=True, shuffle_seed=None,
         crop_area=0.5, crop_area_jitter=0.05,
         crop_aspect_ratio_jitter=0.2,
         do_standardise=True, do_random_flip=True,
         do_class_balance=True):


    steps_per_epoch = 5 # TODO: GET THIS AUTOMAGICALLY FROM DATASET SOMEHOW


    if learning_phases == 1:
        learning_rate = learning_rate_initial
    else:
        learning_rate = optimizers.schedules.PiecewiseConstantDecay(
            [i * int(epochs / learning_phases) * steps_per_epoch for i in range(1, learning_phases)],
            [learning_rate_initial * 10**-i for i in range(learning_phases)])



    # TODO: these were from a *very* small sample - need better estimates
    image_mean = 194.7735
    image_std = 35.388493
    # total = 6265856,  y = 1351927,  not y = 4913929
    # class_weights = n_samples / (n_classes * np.bincount(y))
    class_weights = [0.637560697, 2.317379563]

    # Load dataset
    train_ds = load_dataset(
        "./data/train_image", "./data/train_label",
        shape=input_shape, crop_area=crop_area, crop_area_jitter=crop_area_jitter,
        crop_aspect_ratio_jitter=crop_aspect_ratio_jitter,
        do_standardise=do_standardise, image_mean=image_mean, image_std=image_std,
        do_random_flip=do_random_flip,
        do_shuffle=do_shuffle, shuffle_seed=shuffle_seed,
        do_class_balance=do_class_balance, class_weights=class_weights,
        batch_size=batch_size)

    validate_ds = load_dataset(
        "./data/validate_image", "./data/validate_label",
        shape=input_shape, crop_area=crop_area, crop_area_jitter=crop_area_jitter,
        crop_aspect_ratio_jitter=crop_aspect_ratio_jitter,
        do_standardise=do_standardise, image_mean=image_mean, image_std=image_std,
        do_random_flip=False,
        do_shuffle=False, shuffle_seed=shuffle_seed,
        do_class_balance=do_class_balance, class_weights=class_weights,
        batch_size=batch_size)

    ds = load_dataset(
        "./data/*_image", "./data/*_label",
        shape=(512, 512), crop_area=crop_area, crop_area_jitter=crop_area_jitter,
        crop_aspect_ratio_jitter=crop_aspect_ratio_jitter,
        do_standardise=do_standardise, image_mean=image_mean, image_std=image_std,
        do_random_flip=do_random_flip,
        do_shuffle=False, shuffle_seed=shuffle_seed,
        do_class_balance=True, class_weights=class_weights,
        batch_size=1)






    # # TODO: DEBUG DATASETS

    # #ds = train_ds
    # #ds = validate_ds

    # fig, ax = plt.subplots(2, 6)
    # for i, (x, y, w) in enumerate(ds.take(6)):
    #     print("x", x.shape, x.dtype)
    #     print("y", y.shape, y.dtype)
    #     print("w", w.shape, w.dtype)

    #     xx = x[0].numpy() / 255
    #     yy = y[0].numpy()
    #     ww = w[0].numpy()

    #     ax[0, i].imshow(xx)
    #     ax[1, i].imshow(yy)

    #     #cv2.imshow("test", xx)
    #     #cv2.imshow("test", yy)
    #     #cv2.waitKey(0)
    #     #cv2.destroyAllWindows()

    #     print(xx.shape)
    #     #print(xx)
    #     print(np.min(xx), np.max(xx))
    #     print(yy.shape)
    #     #print(yy)
    #     print(np.unique(yy))
    #     print()
    #     print()

    # #plt.show()
    # exit(0)






    # # TODO: FOR CLASS BALANCING

    # y_size_all = 0
    # y_sum_all = 0
    # y_not_sum_all = 0

    # for x, y in ds:
    #     print(y.shape, np.unique(y))

    #     y_size = np.prod(y.shape)
    #     print("y_size  ", y_size)
    #     y_size_all += y_size

    #     y_not_sum = np.sum(y == 0)
    #     print("ynotsum ", y_not_sum)
    #     y_not_sum_all += y_not_sum

    #     y_sum = np.sum(y == 1)
    #     print("ysum    ", y_sum)
    #     y_sum_all += y_sum

    #     print()

    # print()
    # print("size  ", y_size_all)
    # print("noty  ", y_not_sum_all)
    # print("y     ", y_sum_all)

    # exit(0)





    # === Define Model ===
    # ====================

    # Model inputs
    input_shape = (input_shape[0], input_shape[1], 3)
    inputs = layers.Input(shape=input_shape)

    # Encoder backbone model
    base_model = applications.MobileNetV2(
        input_shape=input_shape, weights="imagenet", include_top=False)

    # Use the outputs of these layers
    base_model_output_names = [
        "block_1_expand_relu",
        "block_3_expand_relu",
        "block_6_expand_relu",
        "block_13_expand_relu",
        "block_16_project",
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in base_model_output_names]

    # Create the downsampling model
    downsample_stack = models.Model(inputs=base_model.input, outputs=base_model_outputs)
    downsample_stack.trainable = False # Do not retrain encoder

    # Downsampling
    downsample_outputs = downsample_stack(inputs)

    # Downsample layers with skip connections
    # (reverse appearance order for upsampling)
    skip_outputs = reversed(downsample_outputs[:-1])

    # Final downsample layer
    x = downsample_outputs[-1]

    def new_upsample_block(filters):
        initializer= tf.random_normal_initializer(0.0, 0.02)

        result = models.Sequential()
        result.add(layers.Conv2DTranspose(
            filters, 3, strides=2, padding="same",
            kernel_initializer=initializer, use_bias=False))
        result.add(layers.BatchNormalization())
        result.add(layers.ReLU())

        return result

    # All upsample blocks
    upsample_blocks = [
        new_upsample_block(512),
        new_upsample_block(256),
        new_upsample_block(128),
        new_upsample_block(64),
    ]

    # Upsampling and skip connections
    for upsample_block, skip_output in zip(upsample_blocks, skip_outputs):
        x = upsample_block(x)
        x = layers.Concatenate()([x, skip_output])

    # Final upsample layer
    x = layers.Conv2DTranspose(n_class, 3, strides=2, padding="same")(x)
    x = layers.Softmax()(x)

    # Create TF model instance
    model = models.Model(inputs=inputs, outputs=x)
    if model_summary:
        model.summary()


    # === Compile & Run Model ===
    # ===========================

    # Configure
    model_optimizer = optimizers.Adam(learning_rate=learning_rate)
    model_loss = losses.SparseCategoricalCrossentropy()

    class MeanIou(metrics.MeanIoU):
        def update_state(self, y_true, y_pred, sample_weight=None):
            y_pred_sparse = tf.argmax(y_pred, axis=-1)
            return super(MeanIou, self).update_state(y_true, y_pred_sparse, sample_weight)

    model_metrics = [MeanIou(n_class), ]

    # Compile and Fit
    #model.compile(optimizer=model_optimizer, loss=model_loss, metrics=model_metrics)
    model.compile(optimizer=model_optimizer, loss=model_loss, weighted_metrics=model_metrics)
    #model.load_weights("test_weights.h5")
    model.load_weights("NEW_weights.h5")
    #model.fit(train_ds, validation_data=validate_ds, epochs=epochs)
    model.evaluate(validate_ds)
    #model.save_weights("test_weights.h5")
    #model.save_weights("NEW_weights.h5")



    #val_next = next(iter(validate_ds))
    for val_next in validate_ds:
        val_x = val_next[0]
        val_y = val_next[1]
        pred_y = model(val_x)

        a = val_x[0].numpy() * image_std + image_mean
        b = val_y[0].numpy()
        c = np.argmax(pred_y[0].numpy(), axis=-1)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(a.astype("uint8"))
        ax[1].imshow(b.astype("uint8"))
        ax[2].imshow(c.astype("uint8"))
        plt.show()







if __name__ == "__main__":

    # Prevent TensorFlow from hogging all GPU memory
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Argument parser
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--learning-rate-initial", type=float, default=0.005)
    parser.add_argument("--learning-phases", type=int, default=3)

    # Dataset parameters
    parser.add_argument("--no-shuffle", dest="do_shuffle", action="store_false")
    parser.add_argument("--no-standardise", dest="do_standardise", action="store_false")
    parser.add_argument("--no-random-flip", dest="do_random_flip", action="store_false")
    parser.add_argument("--no-class-balance", dest="do_class_balance", action="store_false")
    parser.add_argument("--crop-area", type=float, default=0.5)
    parser.add_argument("--crop-area-jitter", type=float, default=0.05)
    parser.add_argument("--crop-aspect-ratio-jitter", type=float, default=0.2)

    # Model parameters
    parser.add_argument("--model-summary", action="store_true")
    parser.add_argument("--input-shape", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--n-class", type=int, default=2)

    # Parse arguments and run
    args = parser.parse_args()
    main(**vars(args))
