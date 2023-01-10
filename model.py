
import tensorflow as tf
from tensorflow.keras import applications, models, layers


def new_upsample_block(filters):
    initializer= tf.random_normal_initializer(0.0, 0.02)

    result = models.Sequential()
    result.add(layers.Conv2DTranspose(
        filters, 3, strides=2, padding="same",
        kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())
    result.add(layers.ReLU())

    return result


def define_model(input_shape, n_class):

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

    return model
