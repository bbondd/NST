import tensorflow as tf
from tensorflow.python import keras as kr
from tensorflow.python.keras.preprocessing import image
import numpy as np

iteration_size = 1000
content_weight = 0.01
style_weight = 1.0

content_path = './pictures/content.jpg'
style_paths = ['./pictures/style1.jpg']

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
                style_loss += style_distance(style_feature, comb_style)

        for target_content, comb_content in zip(content_features, combined_content_features):
            content_loss += tf.reduce_mean(tf.square(comb_content[0], target_content))

        loss = style_loss * style_weight + content_loss * content_weight

    return tape.gradient(loss, combined_array)


def gram_matrix(input_tensor):
    x = tf.reshape(input_tensor, [-1, input_tensor.shape[-1]])
    n = x.shape[0]
    gram = tf.matmul(x, x, transpose_a=True)
    gram / tf.cast(n, tf.float32)

    return tf.reshape(gram, [-1])


def channel_wise_mean_matrix(input_tensor):
    x = tf.reshape(input_tensor, [-1, input_tensor.shape[-1]])
    x = tf.reduce_mean(x, axis=0)

    return x


def euclidean_distance(a, b):
    return tf.reduce_mean(tf.square(style_matrix(a) - style_matrix(b)))


def reversed_euclidean_distance(a, b):
    return -euclidean_distance(a, b)


def minskowki_distance(a, b):
    return tf.reduce_mean(tf.abs(style_matrix(a) - style_matrix(b)))


def js_divergence(a, b):
    a, b = style_matrix(a), style_matrix(b)
    a = tf.reshape(a, [-1]) / tf.reduce_sum(a) + 0.001
    b = tf.reshape(b, [-1]) / tf.reduce_sum(b) + 0.001
    m = (a + b) / 2

    return tf.reduce_mean(a * tf.log(a / m) + b * tf.log(b / m))


def pearson_x2(x, y):
    x, y = style_matrix(x), style_matrix(y)

    x_average = tf.reduce_mean(x)
    y_average = tf.reduce_mean(y)
    x_variance = tf.reduce_mean(tf.square(x - x_average))
    y_variance = tf.reduce_mean(tf.square(y - y_average))

    covariance = tf.reduce_sum((x - x_average) * (y - y_average))

    return -covariance / tf.sqrt(x_variance * y_variance)


def reverse_pearson_x2(x, y):
    return -pearson_x2(x, y)


def squared_hellinger_distance(x, y):
    x, y = style_matrix(x), style_matrix(y)

    x_average = tf.reduce_mean(x)
    y_average = tf.reduce_mean(y)
    x_variance = tf.reduce_mean(tf.square(x - x_average))
    y_variance = tf.reduce_mean(tf.square(y - y_average))
    x_deviation = tf.sqrt(x_variance)
    y_deviation = tf.sqrt(y_variance)

    distance = 1 - tf.sqrt(2 * x_deviation * y_deviation / (x_variance + y_variance)) * \
               tf.exp(-0.25 * tf.square(x_average - y_average) / (x_variance + y_variance))

    return distance * 10000000


def reversed_squared_hellinger_distance(x, y):
    return -squared_hellinger_distance(x, y)


style_matrix = gram_matrix
style_distance = reversed_euclidean_distance


def main():
    model = get_model()

    style_features_group, content_features = get_feature_representations(model)

    combined_array = tf.contrib.eager.Variable(np.expand_dims(content_array, axis=0), dtype=tf.float32)

    for i in range(iteration_size):
        if i % 50 == 0:
            print(i)
            image.save_img(x=image.array_to_img(deprocess_image(combined_array.numpy()[0])),
                           path='./combined/combine%d.jpg' % i)

        gradients = get_gradient(model, combined_array, style_features_group, content_features)
        optimizer.apply_gradients([(gradients, combined_array)])


main()
