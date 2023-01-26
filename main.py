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



    # # TODO: FOR CLASS BALANCING

    # y_size_all = 0
    # y_sum_all = 0
    # y_not_sum_all = 0

    # for x, y in train_ds:
    #     print(y.shape, np.unique(y))

    #     print("y.size  ", y.size)
    #     y_size_all += y.size

    #     y_not_sum = np.sum(y == 0)
    #     print("ynotsum ", y_not_sum)
    #     y_not_sum_all += y_not_sum

    #     y_sum = np.sum(y == 1)
    #     print("ysum    ", y_sum)
    #     y_sum_all += y_sum

    #     print()

    # print()
    # print("size   ", y_size_all)
    # print("y      ", y_sum_all)
    # print("y not  ", y_not_sum_all)

    # exit(0)






    # # TODO: DEBUG DATASETS

    # debug_ds = load_dataset(
    #     "./data/*_image/*", "./data/*_label/*", shape=(512, 512),
    #     do_shuffle=False, shuffle_seed=shuffle_seed,
    #     do_class_balance=True, class_weights=class_weights,
    #     batch_size=1)

    # ds = debug_ds
    # #ds = train_ds
    # #ds = validate_ds

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


    # # === Show Predictions ===
    # # ========================

    # for val_next in validate_ds:
    #     val_x = val_next[0]
    #     val_y = val_next[1]
    #     pred_y = model(val_x)

    #     a = val_x[0].numpy() * image_std + image_mean
    #     b = val_y[0].numpy()
    #     c = np.argmax(pred_y[0].numpy(), axis=2)

    #     fig, ax = plt.subplots(1, 3)
    #     ax[0].imshow(a.astype("uint8"))
    #     ax[1].imshow(b.astype("uint8"))
    #     ax[2].imshow(c.astype("uint8"))
    #     plt.show()

    # exit(0)


    # === Test Large Image ===
    # ========================

    image_path = "./data/test_image/8-1_12-17_TEM_can_it_handle_this_0_0_3072_18432.png"

    import PIL.Image

    with PIL.Image.open(image_path) as image_pil:
        image_raw = np.array(image_pil)

    image = (image_raw - image_mean) / image_std
    image = image[np.newaxis, ...]
    image_raw = image_raw / 255

    with tf.device("/CPU:0"):
        model = define_model(image.shape[1:3], n_class)
        model.load_weights("NEW_weights.h5")
        pred = model(image).numpy()[0]


    # TODO: PREDICTION DECISION BOUNDARY


    # pred_1 = (pred[:, :, 1] > 0.5).astype("uint8")
    # pred_2 = (pred[:, :, 1] > 0.6).astype("uint8")
    # pred_3 = (pred[:, :, 1] > 0.7).astype("uint8")
    # pred_4 = (pred[:, :, 1] > 0.8).astype("uint8")
    # pred_5 = (pred[:, :, 1] > 0.9).astype("uint8")

    # fig, ax = plt.subplots(1, 6, sharex=True, sharey=True)
    # ax[0].imshow(image_raw)
    # ax[1].imshow(pred_1)
    # ax[2].imshow(pred_2)
    # ax[3].imshow(pred_3)
    # ax[4].imshow(pred_4)
    # ax[5].imshow(pred_5)
    # plt.show()


    # pred_1 = (pred[:, :, 1] > 0.90).astype("uint8")
    # pred_2 = (pred[:, :, 1] > 0.91).astype("uint8")
    # pred_3 = (pred[:, :, 1] > 0.92).astype("uint8")
    # pred_4 = (pred[:, :, 1] > 0.93).astype("uint8")
    # pred_5 = (pred[:, :, 1] > 0.94).astype("uint8")
    # pred_6 = (pred[:, :, 1] > 0.95).astype("uint8")
    # pred_7 = (pred[:, :, 1] > 0.96).astype("uint8")

    # fig, ax = plt.subplots(1, 8, sharex=True, sharey=True)
    # ax[0].imshow(image_raw)
    # ax[1].imshow(pred_1)
    # ax[2].imshow(pred_2)
    # ax[3].imshow(pred_3)
    # ax[4].imshow(pred_4)
    # ax[5].imshow(pred_5)
    # ax[6].imshow(pred_6)
    # ax[7].imshow(pred_7)
    # plt.show()


    # TODO: PREDICTION DILATE AND ERODE

    from skimage.draw import disk

    def get_kernel(radius):
        size = 2 * radius + 1
        kernel = np.zeros((size, size), 'uint8')
        r, c = disk((radius, radius), radius)
        kernel[r, c] = 1
        return kernel

    pred_argmax = np.argmax(pred, axis=2).astype("uint8")
    pred_1 = (pred[:, :, 1] > 0.5).astype("uint8")
    pred_2 = (pred[:, :, 1] > 0.6).astype("uint8")
    pred_3 = (pred[:, :, 1] > 0.7).astype("uint8")
    pred_4 = (pred[:, :, 1] > 0.8).astype("uint8")
    pred_5 = (pred[:, :, 1] > 0.9).astype("uint8")

    pred_1_processed_0 = pred_1
    #pred_2_processed_0 = pred_2
    #pred_3_processed_0 = pred_3
    #pred_4_processed_0 = pred_4
    #pred_5_processed_0 = pred_5

    kernel_5 = get_kernel(5)
    kernel_10 = get_kernel(10)
    kernel_15 = get_kernel(15)

    pred_1_processed_1 = cv2.erode(pred_1_processed_0, kernel_5)
    pred_1_processed_1 = cv2.dilate(pred_1_processed_1, kernel_5)
    pred_1_processed_1 = cv2.dilate(pred_1_processed_1, kernel_5)
    pred_1_processed_1 = cv2.erode(pred_1_processed_1, kernel_5)

    pred_1_processed_2 = cv2.erode(pred_1_processed_0, kernel_10)
    pred_1_processed_2 = cv2.dilate(pred_1_processed_2, kernel_10)
    pred_1_processed_2 = cv2.dilate(pred_1_processed_2, kernel_10)
    pred_1_processed_2 = cv2.erode(pred_1_processed_2, kernel_10)

    pred_1_processed_3 = cv2.erode(pred_1_processed_0, kernel_15)
    pred_1_processed_3 = cv2.dilate(pred_1_processed_3, kernel_15)
    pred_1_processed_3 = cv2.dilate(pred_1_processed_3, kernel_15)
    pred_1_processed_3 = cv2.erode(pred_1_processed_3, kernel_15)

    # fig, ax = plt.subplots(1, 5, sharex=True, sharey=True)
    # ax[0].imshow(image_raw)
    # ax[1].imshow(pred_1_processed_0)
    # ax[2].imshow(pred_1_processed_1)
    # ax[3].imshow(pred_1_processed_2)
    # ax[4].imshow(pred_1_processed_3)
    # plt.show()


    im = image_raw.copy()
    mask_1 = pred_1_processed_1
    mask_2 = pred_1_processed_2
    mask_3 = pred_1_processed_3

    # # Slice for plot (optional)
    # r = 2448
    # c = 1424
    # s = 400
    # im = im[r:r+s, c:c+s]
    # mask_1 = mask_1[r:r+s, c:c+s]
    # mask_2 = mask_2[r:r+s, c:c+s]
    # mask_3 = mask_3[r:r+s, c:c+s]

    # Find segmentation contours
    contours_1, hierarchy_1 = cv2.findContours(mask_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_2, hierarchy_2 = cv2.findContours(mask_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_3, hierarchy_3 = cv2.findContours(mask_3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Segment perimeter
    #for i, cnt in enumerate(contours_1):
    #    mask_1_perimeter = cv2.arcLength(cnt, True)
    #    print(f"mask_1_perimeter_{i}: {mask_1_perimeter}")
    mask_1_total_perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours_1)
    print(f"mask_1_total_perimeter: {mask_1_total_perimeter}")
    #for i, cnt in enumerate(contours_2):
    #    mask_2_perimeter = cv2.arcLength(cnt, True)
    #    print(f"mask_2_perimeter_{i}: {mask_2_perimeter}")
    mask_2_total_perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours_2)
    print(f"mask_2_total_perimeter: {mask_2_total_perimeter}")
    #for i, cnt in enumerate(contours_3):
    #    mask_3_perimeter = cv2.arcLength(cnt, True)
    #    print(f"mask_3_perimeter_{i}: {mask_3_perimeter}")
    mask_3_total_perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours_3)
    print(f"mask_3_total_perimeter: {mask_3_total_perimeter}")

    # Segment area
    mask_1_size = mask_1.size
    print(f"mask_1_size {mask_1_size}")
    mask_1_area = mask_1.sum()
    print(f"mask_1_area {mask_1_area}")
    mask_2_size = mask_2.size
    #print(f"mask_2_size {mask_2_size}")
    mask_2_area = mask_2.sum()
    print(f"mask_2_area {mask_2_area}")
    mask_3_size = mask_3.size
    #print(f"mask_3_size {mask_3_size}")
    mask_3_area = mask_3.sum()
    print(f"mask_3_area {mask_3_area}")

    # Segment density
    mask_1_density = mask_1_area / mask_1_total_perimeter
    print(f"mask_1_density {mask_1_density}")
    mask_2_density = mask_2_area / mask_2_total_perimeter
    print(f"mask_2_density {mask_2_density}")
    mask_3_density = mask_3_area / mask_3_total_perimeter
    print(f"mask_3_density {mask_3_density}")

    # Draw segmentation boundaries
    #cv2.drawContours(im, contours_1, -1, (0.0, 0.0, 1.0), 3)
    #cv2.drawContours(im, contours_2, -1, (0.0, 0.0, 1.0), 3)
    cv2.drawContours(im, contours_3, -1, (0.0, 0.0, 1.0), 3)
    #cv2.drawContours(im, [contours_1[2]], 0, (0.0, 0.0, 1.0), 3)
    #cv2.drawContours(im, [contours_2[2]], 0, (0.0, 0.0, 1.0), 3)
    #cv2.drawContours(im, [contours_3[2]], 0, (0.0, 0.0, 1.0), 3)

    # Plot
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True)
    ax[0].imshow(im)
    ax[1].imshow(mask_1)
    ax[2].imshow(mask_2)
    ax[3].imshow(mask_3)
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
