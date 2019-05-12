import numpy as np
import keras as k
import keras.backend as b
from keras.preprocessing import image


content_array = image.img_to_array(image.load_img('./pictures/content.jpg')) / 255
style_array = image.img_to_array(image.load_img('./pictures/style.jpg')) / 255

epoch = 100000

combine_shape = content_array.shape
"""
content_output_names = [
    'block5_conv4',
    'block5_conv3',
    'block5_conv2',
    'block5_conv1',
    'block4_conv4',
    'block4_conv3',
    'block4_conv2',
    'block4_conv1',
    'block3_conv4',
    'block3_conv3',
    'block3_conv2',
    'block3_conv1',
    'block2_conv2',
    'block2_conv1',
    'block1_conv2',
    'block1_conv1',
]
style_output_names = [
    'block5_conv4',
    'block5_conv3',
]"""
content_output_names = [
    'block5_conv2',
    'block5_conv1',

]
style_output_names = [
    'block4_conv1',
    'block3_conv1',
    'block2_conv1',
]
content_loss_weight = 0.03
style_loss_weight = 1


def gram_matrix(x):
    width = int(x.shape[1])
    height = int(x.shape[2])
    channel_size = int(x.shape[3])

    x = b.permute_dimensions(x, [0, 3, 1, 2])
    x = b.reshape(x, [-1, channel_size, width * height])
    x = b.batch_dot(x, b.permute_dimensions(x, [0, 2, 1]))
    x = x / (width * height * channel_size)

    return x


def make_models():
    combine_input = k.Input(shape=[1])
    combine_picture_layer = k.layers.Dense(
        units=combine_shape[0] * combine_shape[1] * combine_shape[2],
        activation='linear',
    )(combine_input)
    combine_picture_layer = k.layers.Reshape(
        target_shape=combine_shape
    )(combine_picture_layer)

    style_layer = style_input = k.Input(shape=style_array.shape)
    content_layer = content_input = k.Input(shape=content_array.shape)

    combine_layer = combine_picture_layer
    combine_style_outputs = []
    combine_content_outputs = []
    style_outputs = []
    content_outputs = []

    vgg = k.applications.VGG19(weights='imagenet', include_top=False)

    for layer in vgg.layers:
        layer.trainable = False
        combine_layer = layer(combine_layer)
        content_layer = layer(content_layer)
        style_layer = layer(style_layer)

        if layer.name in content_output_names:
            combine_content_outputs.append(combine_layer)
            content_outputs.append(content_layer)

        if layer.name in style_output_names:
            combine_style_outputs.append(
                k.layers.Lambda(gram_matrix)(combine_layer)
            )
            style_outputs.append(
                k.layers.Lambda(gram_matrix)(style_layer)
            )

    content_loss_layers = [k.layers.subtract([combine_content_output, content_output])
                           for combine_content_output, content_output
                           in zip(combine_content_outputs, content_outputs)]
    content_loss_layers = [k.layers.Lambda(lambda x: b.reshape(b.sum(b.square(x), axis=[1, 2, 3]), [-1, 1]))(content_loss_layer)
                           for content_loss_layer
                           in content_loss_layers]
    if len(content_loss_layers) == 1:
        content_loss_layer = content_loss_layers[0]
    else:
        content_loss_layer = k.layers.add(content_loss_layers)

    style_loss_layers = [k.layers.subtract([combine_style_output, style_output])
                         for combine_style_output, style_output
                         in zip(combine_style_outputs, style_outputs)]
    style_loss_layers = [k.layers.Lambda(lambda x: b.reshape(b.sum(b.square(x), axis=[1, 2]), [-1, 1]))(style_loss_layer)
                         for style_loss_layer
                         in style_loss_layers]
    if len(style_loss_layers) == 1:
        style_loss_layer = style_loss_layers[0]
    else:
        style_loss_layer = k.layers.add(style_loss_layers)

    total_loss_layer = k.layers.Lambda(
        lambda x: x[0] * content_loss_weight + x[1] * style_loss_weight
    )([content_loss_layer, style_loss_layer])

    train_model = k.Model(inputs=[content_input, style_input, combine_input], outputs=[total_loss_layer])

    train_model.compile(optimizer=k.optimizers.Adam(lr=0.03), loss='mean_squared_error')

    picture_model = k.Model(inputs=[combine_input], outputs=[combine_picture_layer])

    return train_model, picture_model


def main():
    train_model, picture_model = make_models()

    for i in range(epoch):
        train_model.fit(x=[np.array([content_array]), np.array([style_array]), np.array([[1]])],
                        y=[np.array([[0]])], verbose=0, epochs=49)
        train_model.fit(x=[np.array([content_array]), np.array([style_array]), np.array([[1]])],
                        y=[np.array([[0]])])
        print(i * 50)

        picture_array = picture_model.predict(x=[np.array([[1]])])[0] * 255
        #picture_array = np.array(picture_array, dtype='uint8')

        image.save_img(x=image.array_to_img(picture_array), path='./combined/combine %d.jpg' % i)


main()
