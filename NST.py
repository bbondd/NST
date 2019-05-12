from PIL import Image
import numpy as np
import keras as k
import keras.backend as b

content_array = np.array(Image.open('./pictures/starry_night.jpg')) / 255
style_array = np.array(Image.open('./pictures/sunflower.jpg')) / 255
epoch = 100000


combine_shape = content_array.shape


def gram_matrix(x):
    x = b.permute_dimensions(x, [0, 3, 1, 2])
    x = b.reshape(x, [-1, x.shape[1], x.shape[2] * x.shape[3]])
    x = b.batch_dot(x, b.permute_dimensions(x, [0, 2, 1]))

    return x


def make_models():
    combine_input = k.Input(shape=[1])
    combine_picture_layer = k.layers.Dense(
        units=combine_shape[0] * combine_shape[1] * combine_shape[2],
        activation='sigmoid',
        use_bias=False,
    )(combine_input)
    combine_picture_layer = k.layers.Reshape(
        target_shape=combine_shape
    )(combine_picture_layer)

    style_layer = style_input = k.Input(shape=style_array.shape)
    content_layer = content_input = k.Input(shape=content_array.shape)

    combine_layer = combine_picture_layer
    combine_style_output = None
    combine_content_output = None
    style_output = None
    content_output = None

    vgg = k.applications.VGG19(weights='imagenet', include_top=False)
    vgg.trainable = False

    for layer in vgg.layers:
        combine_layer = layer(combine_layer)
        style_layer = layer(style_layer)
        content_layer = layer(content_layer)

        if layer.name == 'block5_conv4':
            combine_content_output = combine_layer
            content_output = content_layer

        if layer.name == 'block4_conv4':
            combine_style_output = k.layers.Lambda(gram_matrix)(combine_layer)
            style_output = k.layers.Lambda(gram_matrix)(style_layer)

    content_loss_layer = k.layers.subtract([combine_content_output, content_output])
    content_loss_layer = k.layers.Lambda(lambda x: b.reshape(b.sum(b.square(x), axis=[1, 2, 3]), [-1, 1]))(content_loss_layer)

    style_loss_layer = k.layers.subtract([combine_style_output, style_output])
    style_loss_layer = k.layers.Lambda(lambda x: b.reshape(b.sum(b.square(x), axis=[1, 2]), [-1, 1]))(style_loss_layer)

    total_loss_layer = k.layers.add([style_loss_layer, content_loss_layer])

    train_model = k.Model(inputs=[content_input, style_input, combine_input], outputs=[total_loss_layer])
    k.utils.plot_model(train_model, show_shapes=True)

    train_model.compile(optimizer=k.optimizers.Adam(lr=0.1), loss='mean_squared_error')

    picture_model = k.Model(inputs=[combine_input], outputs=[combine_picture_layer])

    return train_model, picture_model


def main():
    train_model, picture_model = make_models()

    for i in range(epoch):
        if i % 10 != 0:
            train_model.fit(x=[np.array([content_array]), np.array([style_array]), np.array([[1]])],
                            y=[np.array([[0]])], verbose=0)

        elif i % 10 == 0:
            print(i)
            train_model.fit(x=[np.array([content_array]), np.array([style_array]), np.array([[1]])],
                            y=[np.array([[0]])])

        if i % 50 == 0:
            picture_array = picture_model.predict(x=[np.array([[1]])])[0] * 255
            Image.fromarray(picture_array, 'RGB').save('./pictures/combine %d.jpg' % i)


main()
