import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python import keras as kr
from tensorflow.python.keras.preprocessing import image
import numpy as np

tf.enable_eager_execution()

content_path = './pictures/content.jpg'
style_path = './pictures/style.jpg'

content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1',]


def load_and_process_image(path):
    image_array = image.img_to_array(image.load_img(path))
    image_array = kr.applications.vgg19.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)


content_array = load_and_process_image(content_path)
style_array = load_and_process_image(style_path)


style_weight = 1.0
content_weight = 0.01
iteration_size = 500


def deprocess_image(image_array):
    x = image_array[0]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_model():
    vgg = kr.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs

    return kr.Model(vgg.input, model_outputs)


def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    x = tf.reshape(input_tensor, [-1, input_tensor.shape[-1]])
    n = x.shape[0]
    gram = tf.matmul(x, x, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    return tf.reduce_mean(tf.square(gram_matrix(base_style) - gram_target))


def get_feature_representations(model):
    style_outputs = model(style_array)
    content_outputs = model(content_array)

    style_features = [style_layer[0] for style_layer in style_outputs[:len(style_layers)]]
    content_features = [content_layer[0] for content_layer in content_outputs[len(style_layers):]]

    return style_features, content_features


def compute_loss(model, init_image, gram_style_features, content_features):
    model_outputs = model(init_image)

    style_output_features = model_outputs[:len(style_layers)]
    content_output_features = model_outputs[len(style_layers):]

    style_score, content_score = 0, 0

    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += get_style_loss(comb_style[0], target_style)

    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score

    return loss, style_score, content_score


def compute_grads(model, init_image, gram_style_features, content_features):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(model, init_image, gram_style_features, content_features)

    total_loss = all_loss[0]

    return tape.gradient(total_loss, init_image), all_loss


def run_style_transfer():
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    style_features, content_features = get_feature_representations(model)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    init_image = tfe.Variable(content_array, dtype=tf.float32)

    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

    loss_weights = (style_weight, content_weight)

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    for _ in range(iteration_size):
        grades, all_loss = compute_grads(model, init_image, gram_style_features, content_features)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grades, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

    return deprocess_image(init_image.numpy())


combined_array = run_style_transfer()
image.save_img(x=image.array_to_img(combined_array), path='./combined/combine.jpg')

