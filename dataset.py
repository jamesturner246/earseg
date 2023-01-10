import tensorflow as tf
import tensorflow_io as tfio


def load_png(image_path, label_path):
    # Read files into memory
    image = tf.io.read_file(image_path)
    label = tf.io.read_file(label_path)

    # Decode PNG format files
    image = tf.image.decode_png(image)
    label = tf.image.decode_png(label)

    # Check image and label mask shapes match
    tf.debugging.assert_equal(
        tf.shape(image), tf.shape(label),
        message="image and label mask shape mismatch")

    # Prepare image
    image = tf.cast(image, tf.float32)

    # Prepare label mask
    # NOTE: assumes pos is (255, 0, 0) and neg is (255, 255, 255)
    label = label[:, :, 1:2]
    label = tf.where(label > 0, tf.constant(0, dtype=tf.uint8), tf.constant(1, dtype=tf.uint8))

    return image, label


def load_tiff(image_path, label_path):
    # Read files into memory
    image = tf.io.read_file(image_path)
    label = tf.io.read_file(label_path)

    # Decode TIFF format files
    image = tfio.experimental.image.decode_tiff(image)
    label = tfio.experimental.image.decode_tiff(label)

    # Check image and label mask shapes match
    tf.debugging.assert_equal(
        tf.shape(image), tf.shape(label),
        message="image and label mask shape mismatch")

    # Prepare image
    image = image[:, :, 0:3]
    image = tf.cast(image, tf.float32)

    # Prepare label mask
    # NOTE: assumes pos is (255, 0, 0) and neg is (255, 255, 255)
    label = label[:, :, 1:2]
    label = tf.where(label > 0, tf.constant(0, dtype=tf.uint8), tf.constant(1, dtype=tf.uint8))

    return image, label


def resize(shape):
    # Resize image
    # input (height, width, channels)

    def fn(image, label):
        image = tf.image.resize(image, shape, method='bilinear')
        label = tf.image.resize(label, shape, method='nearest')
        return image, label

    return fn


def standardise(mean, std):
    # samplewise standardise (scalars) or colourwise standardise (3-vectors)
    # input (height, width, channels) or (batch, height, width, channels)

    def fn(image, label):
        image = tf.math.subtract(image, mean)
        image = tf.math.divide(image, std)
        return image, label

    return fn


def random_flip():
    # Flip image left to right with 0.5 probability
    # and top to bottom with 0.5 probability
    # input (height, width, channels)

    def fn(image, label):
        r = tf.random.uniform((2,)) < 0.5
        if r[0]:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
        if r[1]:
            image = tf.image.flip_up_down(image)
            label = tf.image.flip_up_down(label)
        return image, label

    return fn


def random_crop(
        shape, crop_area=0.5, crop_area_jitter=0.05,
        crop_aspect_ratio_jitter=0.2):
    # Random crop image
    # input (height, width, channels)

    crop_area_min = crop_area - crop_area_jitter
    crop_area_max = crop_area + crop_area_jitter
    crop_aspect_ratio_min = 1.0 - crop_aspect_ratio_jitter
    crop_aspect_ratio_max = 1.0 + crop_aspect_ratio_jitter

    def fn(image, label):
        image_shape = tf.shape(image)

        # Find a bbox
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            image_shape, tf.zeros((0, 0, 4)), min_object_covered=0.0,
            aspect_ratio_range=[crop_aspect_ratio_min, crop_aspect_ratio_max],
            area_range=[crop_area_min, crop_area_max],
            max_attempts=100, use_image_if_no_bounding_boxes=True)

        # Crop image with bbox
        image = tf.slice(image, bbox_begin, bbox_size)
        label = tf.slice(label, bbox_begin, bbox_size)
        return image, label

    return fn


def sample_weights(class_weights):
    # Adds loss weighting image for sample
    # input (height, width, channels) or (batch, height, width, channels)

    # Class weights, such that sum(class_weights) == 1.0
    class_weights = tf.constant(class_weights, dtype=tf.float32)
    class_weights /= tf.reduce_sum(class_weights)

    def fn(image, label):
        # Create sample weights image
        sample_weight = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

        return image, label, sample_weight

    return fn


def load_dataset(
        image_path_pattern, label_path_pattern, shape=None,
        do_crop=False, crop_area=0.5, crop_area_jitter=0.05, crop_aspect_ratio_jitter=0.2,
        do_standardise=False, image_mean=None, image_std=None,
        do_random_flip=False,
        do_shuffle=False, shuffle_seed=None,
        do_class_balance=False, class_weights=None,
        batch_size=1):

    # Find dataset files
    image_files_ds = tf.data.Dataset.list_files(image_path_pattern, shuffle=False)
    label_files_ds = tf.data.Dataset.list_files(label_path_pattern, shuffle=False)
    files_ds = tf.data.Dataset.zip((image_files_ds, label_files_ds))
    files_ds = files_ds.cache()

    # Shuffle
    if do_shuffle:
        print("preprocessing: shuffle")
        files_ds = files_ds.shuffle(1000, seed=shuffle_seed)

    # Load images and labels
    print("preprocessing: load_image")
    ds = files_ds.map(load_png, num_parallel_calls=tf.data.AUTOTUNE)
    #ds = files_ds.map(load_tiff, num_parallel_calls=tf.data.AUTOTUNE)

    # Random cropping
    if do_crop:
        print("preprocessing: random_crop")
        random_crop_fn = random_crop(
            shape, crop_area=crop_area, crop_area_jitter=crop_area_jitter,
            crop_aspect_ratio_jitter=crop_aspect_ratio_jitter)
        ds = ds.map(random_crop_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Resize image
    if shape is not None:
        print("preprocessing: resize")
        resize_fn = resize(shape)
        ds = ds.map(resize_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Standardise image
    if do_standardise:
        print("preprocessing: standardise")
        standardise_fn = standardise(image_mean, image_std)
        ds = ds.map(standardise_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Random flipping
    if do_random_flip:
        print("preprocessing: random_flip")
        random_flip_fn = random_flip()
        ds = ds.map(random_flip_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Add sample weights image
    if do_class_balance:
        print("preprocessing: sample_weights")
        sample_weights_fn = sample_weights(class_weights)
        ds = ds.map(sample_weights_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
