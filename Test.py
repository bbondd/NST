import tensorflow as tf
from tensorflow.python import keras as kr
from tensorflow.python.keras.preprocessing import image
import numpy as np
import sys


# python TFNST.py iteration_size content_weight style_weight content_path style_path0 style_path1...
iteration_size = 500
content_weight = 0.01
style_weight = 1.0
content_path = './pictures/content.jpg'
style_paths = ['./pictures/style2.jpg']

style_weight /= len(style_paths)

tf.enable_eager_execution()


def load_and_process_image(path):
    image_array = image.img_to_array(image.load_img(path))
    return kr.applications.vgg19.preprocess_input(image_array)


content_array = load_and_process_image(content_path)
style_arrays = [load_and_process_image(style_path) for style_path in style_paths]


content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1',]

optimizer = tf.train.AdamOptimizer(learning_rate=3, beta1=0.99, epsilon=1e-1)


def deprocess_image(image_array):
    x = image_array
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_model():
    vgg = kr.applications.VGG19(include_top=False, weights='imagenet')

    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs

    model = kr.Model(vgg.input, model_outputs)

    for layer in model.layers:
        layer.trainable = False

    return model


def gram_matrix(input_tensor):
    x = tf.reshape(input_tensor, [-1, input_tensor.shape[-1]])
    n = x.shape[0]
    gram = tf.matmul(x, x, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_feature_representations(model):
    style_features_group = []
    for style_array in style_arrays:
        style_outputs = model(np.expand_dims(style_array, axis=0))
        style_features_group.append([style_layer[0] for style_layer in style_outputs[:len(style_layers)]])

    content_outputs = model(np.expand_dims(content_array, axis=0))

    content_features = [content_layer[0] for content_layer in content_outputs[len(style_layers):]]

    return style_features_group, content_features


def get_gradient(model, combined_array, style_features_group, content_features):
    with tf.GradientTape() as tape:
        model_outputs = model(combined_array)

        combined_style_features = model_outputs[:len(style_layers)]
        combined_content_features = model_outputs[len(style_layers):]

        style_loss, content_loss = 0, 0

        for style_features in style_features_group:
            for style_feature, comb_style in zip(style_features, combined_style_features):
                style_loss += tf.reduce_mean(tf.square(gram_matrix(style_feature) - gram_matrix(comb_style)))

        for target_content, comb_content in zip(content_features, combined_content_features):
            content_loss += tf.reduce_mean(tf.square(comb_content[0], target_content))

        loss = style_loss * style_weight + content_loss * content_weight

    return tape.gradient(loss, combined_array)


def main():
    model = get_model()

    style_features_group, content_features = get_feature_representations(model)

    combined_array = tf.contrib.eager.Variable(np.expand_dims(content_array, axis=0), dtype=tf.float32)

    for i in range(iteration_size):
        gradients = get_gradient(model, combined_array, style_features_group, content_features)
        optimizer.apply_gradients([(gradients, combined_array)])

        if i % 50 == 0:
            image.save_img(x=image.array_to_img(deprocess_image(combined_array.numpy()[0])),
                           path='./combined/combine%d.jpg' % i)


main()
