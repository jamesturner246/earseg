import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics

from dataset import *
from model import *


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

    # # Load dataset
    # train_ds = load_dataset(
    #     "./data/train_image/*", "./data/train_label/*", shape=input_shape,
    #     do_crop=True, crop_area=crop_area, crop_area_jitter=crop_area_jitter,
    #     crop_aspect_ratio_jitter=crop_aspect_ratio_jitter,
    #     do_standardise=do_standardise, image_mean=image_mean, image_std=image_std,
    #     do_random_flip=do_random_flip,
    #     do_shuffle=do_shuffle, shuffle_seed=shuffle_seed,
    #     do_class_balance=do_class_balance, class_weights=class_weights,
    #     batch_size=batch_size)

    # validate_ds = load_dataset(
    #     "./data/validate_image/*", "./data/validate_label/*", shape=input_shape,
    #     do_standardise=do_standardise, image_mean=image_mean, image_std=image_std,
    #     do_random_flip=False,
    #     do_shuffle=False, shuffle_seed=shuffle_seed,
    #     do_class_balance=do_class_balance, class_weights=class_weights,
    #     batch_size=batch_size)

    # debug_ds = load_dataset(
    #     "./data/*_image/*", "./data/*_label/*", shape=(512, 512),
    #     do_shuffle=False, shuffle_seed=shuffle_seed,
    #     do_class_balance=True, class_weights=class_weights,
    #     batch_size=1)






    # # TODO: DEBUG DATASETS

    # #ds = train_ds
    # #ds = validate_ds
    # ds = debug_ds

    # fig, ax = plt.subplots(2, 6)
    # for i, (x, y, w) in enumerate(ds.take(6)):
    #     xx = x[0].numpy() / 255
    #     yy = y[0].numpy()
    #     ww = w[0].numpy()

    #     print(f"x {xx.dtype} {xx.shape}")
    #     print(f"y {yy.dtype} {yy.shape}")
    #     print(f"w {ww.dtype} {ww.shape}")
    #     print()
    #     print(f"x range: [{np.min(xx)}, {np.max(xx)}]")
    #     print(f"y unique: {np.unique(yy)}")
    #     print()
    #     print()

    #     ax[0, i].imshow(xx)
    #     ax[1, i].imshow(yy)

    #     cv2.imshow("x", xx.astype("float32"))
    #     cv2.imshow("y", yy.astype("float32"))
    #     cv2.waitKey(0)

    # plt.show()
    # cv2.destroyAllWindows()
    # exit(0)






    # # TODO: FOR CLASS BALANCING

    # y_size_all = 0
    # y_sum_all = 0
    # y_not_sum_all = 0

    # for x, y in train_ds:
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








    # # === Compile & Run Model ===
    # # ===========================

    # model = define_model(input_shape, n_class)
    # if model_summary:
    #     model.summary()

    # # Configure
    # model_optimizer = optimizers.Adam(learning_rate=learning_rate)
    # model_loss = losses.SparseCategoricalCrossentropy()
    # model_metrics = [metrics.MeanIoU(n_class, sparse_y_pred=False), ]

    # # Compile and Fit
    # #model.compile(optimizer=model_optimizer, loss=model_loss, metrics=model_metrics)
    # model.compile(optimizer=model_optimizer, loss=model_loss, weighted_metrics=model_metrics)
    # #model.load_weights("test_weights.h5")
    # model.load_weights("NEW_weights.h5")
    # #model.fit(train_ds, validation_data=validate_ds, epochs=epochs)
    # model.evaluate(validate_ds)
    # #model.save_weights("test_weights.h5")
    # #model.save_weights("NEW_weights.h5")





    # for val_next in validate_ds:
    #     val_x = val_next[0]
    #     val_y = val_next[1]
    #     pred_y = model(val_x)

    #     a = val_x[0].numpy() * image_std + image_mean
    #     b = val_y[0].numpy()
    #     c = np.argmax(pred_y[0].numpy(), axis=-1)

    #     fig, ax = plt.subplots(1, 3)
    #     ax[0].imshow(a.astype("uint8"))
    #     ax[1].imshow(b.astype("uint8"))
    #     ax[2].imshow(c.astype("uint8"))
    #     plt.show()

    





    # === Test Large Image ===
    # ========================


    import PIL.Image

    image_path = "./data/test_image/8-1_12-17_TEM_can_it_handle_this_0_0_3072_18432.png"

    with PIL.Image.open(image_path) as image_pil:

        #new_scale = 1
        new_scale = 7/8
        #new_scale = 3/4

        new_shape = tuple(int(s * new_scale) for s in image_pil.size)
        image_raw = np.array(image_pil.resize(new_shape))

    image = (image_raw - image_mean) / image_std
    image = image[np.newaxis, ...]
    image_raw = image_raw / 255


    model = define_model(image.shape[1:3], n_class)
    model.load_weights("NEW_weights.h5")
    pred = model(image).numpy()[0]
    pred = np.argmax(pred, axis=2)


    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(image_raw)
    ax[1].imshow(pred)
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
