from tensorflow import math
from tensorflow.keras.layers import Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Input

from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.layers import GaussianNoise
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import UpSampling1D


class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * math.tanh(math.softplus(inputs))


def noised_ae(input_shape, noise=False, normalization=False):
    CONV_SIZES = 16, 32, 128  # filters for obj/subj (0, 1) & large filter for large predicate (2)
    INNER_SIZE = CONV_SIZES[0] * 2 + CONV_SIZES[1]

    def encode_embedding_input(input_layer, large=False):
        conv1_size, conv2_size = (CONV_SIZES[2], CONV_SIZES[1]) if large else (CONV_SIZES[1], CONV_SIZES[0])

        if noise:
            input_layer = GaussianNoise(stddev=.001)(input_layer)
        
        if normalization:
            input_layer = BatchNormalization()(input_layer)
        
        conv1 = Conv1D(conv1_size, (2,), activation='mish', padding='same')(input_layer)
        pool1 = MaxPooling1D((2,), padding='same')(conv1)
        conv2 = Conv1D(conv2_size, (2,), activation='mish', padding='same')(pool1)
        pool2 = MaxPooling1D((2,), padding='same')(conv2)
        return Flatten()(pool2)

    def decode_embedding_input(latent, shape, name, large=False):
        conv1_size = 128 if large else 64

        latent = Reshape((1, INNER_SIZE))(latent)
        conv1 = Conv1D(conv1_size, (1,), activation='mish', padding='same', name=name + '_conv1')(latent)
        up1 = UpSampling1D(shape[0], name=name + '_up1')(conv1)
        conv2 = Conv1D(shape[1], (6,), activation='mish', padding='same', name=name + '_conv2')(up1)
        return conv2

    input_subject = Input(shape=input_shape[0], name='input_subject')
    input_object = Input(shape=input_shape[1], name='input_object')
    input_rel = Input(shape=input_shape[2], name='input_rel')

    encode_subject = encode_embedding_input(input_subject)
    encode_object = encode_embedding_input(input_object)
    encode_rel = encode_embedding_input(input_rel, large=True)

    latent = concatenate([encode_subject, encode_object, encode_rel], name='embedding')
    #     latent = Dense(INNER_SIZE, activation='sigmoid', name='embedding')(x)

    output_subject = decode_embedding_input(latent, shape=input_shape[0], name='output_subject')
    output_object = decode_embedding_input(latent, shape=input_shape[1], name='output_object')
    output_rel = decode_embedding_input(latent, shape=input_shape[2], name='output_rel', large=True)

    model = Model(inputs=[input_subject, input_object, input_rel],
                  outputs=[output_subject, output_object, output_rel])

    return model


def masked_ae(input_shape, large=False):
    """ mask relation embedding and try to restore it along with the arguments """

    CONV_SIZES = 16, 32, 128  # filters for obj/subj (0, 1) & large filter for large predicate (2)
    INNER_SIZE = CONV_SIZES[0] * 2 + CONV_SIZES[1]

    def encode_embedding_input(input_layer):
        conv1_size, conv2_size = (CONV_SIZES[2], CONV_SIZES[1])

        input_layer = GaussianNoise(stddev=.1)(input_layer)
        conv1 = Conv1D(conv1_size, (2,), activation='mish', padding='same')(input_layer)
        pool1 = MaxPooling1D((2,), padding='same')(conv1)
        conv2 = Conv1D(conv2_size, (2,), activation='mish', padding='same')(pool1)
        pool2 = MaxPooling1D((2,), padding='same')(conv2)
        return Flatten()(pool2)

    def decode_embedding_input(latent, shape, name):
        conv1_size = 128

        latent = Reshape((1, INNER_SIZE))(latent)
        conv1 = Conv1D(conv1_size, (1,), activation='mish', padding='same', name=name + '_conv1')(latent)
        up1 = UpSampling1D(shape[0], name=name + '_up1')(conv1)
        conv2 = Conv1D(shape[1], (8,), activation='mish', padding='same', name=name + '_conv2')(up1)
        return conv2

    input_subject = Input(shape=input_shape[0], name='input_subject')
    input_object = Input(shape=input_shape[1], name='input_object')
    input_rel = Input(shape=input_shape[2], name='input_rel')

    encode_subject = encode_embedding_input(input_subject)
    encode_object = encode_embedding_input(input_object)

    latent = concatenate([encode_subject, encode_object], name='embedding')

    output_subject = decode_embedding_input(latent, shape=input_shape[0], name='output_subject')
    output_object = decode_embedding_input(latent, shape=input_shape[1], name='output_object')
    output_rel = decode_embedding_input(latent, shape=input_shape[2], name='output_rel')

    model = Model(inputs=[input_subject, input_object, input_rel],
                  outputs=[output_subject, output_object, output_rel])

    return model


def restore_rel(input_shape):
    """ mask relation embedding and try to restore it alone """

    INNER_SIZE = 100

    def encode_embedding_input(input_layer):
        conv1 = Conv1D(128, (2,), activation='relu', padding='same')(input_layer)
        pool1 = MaxPooling1D((2,), padding='same')(conv1)
        conv2 = Conv1D(64, (2,), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling1D((2,), padding='same')(conv2)
        return Flatten()(pool2)

    def decode_embedding_input(latent, name):
        latent = Reshape((1, INNER_SIZE))(latent)
        conv1 = Conv1D(128, (1,), activation='relu', padding='same', name=name + '_conv1')(latent)
        up1 = UpSampling1D(input_shape[0], name=name + '_up1')(conv1)
        conv2 = Conv1D(input_shape[1], (6,), activation='relu', padding='same', name=name + '_conv2')(up1)
        return conv2

    input_subject = Input(shape=input_shape[0], name='input_subject')
    input_sub_noised = GaussianNoise(stddev=.001)(input_subject)
    input_object = Input(shape=input_shape[1], name='input_object')
    input_obj_noised = GaussianNoise(stddev=.001)(input_object)
    input_rel = Input(shape=input_shape[2], name='input_rel')

    encode_subject = encode_embedding_input(input_subject)
    encode_object = encode_embedding_input(input_object)

    x = concatenate([encode_subject, encode_object])
    latent = Dense(INNER_SIZE, activation='sigmoid', name='embedding')(x)

    output_rel = decode_embedding_input(latent, 'output_rel')

    model = Model(inputs=[input_subject, input_object, input_rel],
                  outputs=[input_sub_noised, input_obj_noised, output_rel])

    return model


def restore_obj(input_shape):
    """ mask object embedding and try to restore it alone """

    INNER_SIZE = 50

    def encode_embedding_input(input_layer):
        conv1 = Conv1D(128, (2,), activation='relu', padding='same')(input_layer)
        pool1 = MaxPooling1D((2,), padding='same')(conv1)
        conv2 = Conv1D(32, (2,), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling1D((2,), padding='same')(conv2)
        return Flatten()(pool2)

    def decode_embedding_input(latent, name):
        latent = Reshape((1, INNER_SIZE))(latent)
        conv1 = Conv1D(128, (1,), activation='relu', padding='same', name=name + '_conv1')(latent)
        up1 = UpSampling1D(input_shape[0], name=name + '_up1')(conv1)
        conv2 = Conv1D(input_shape[1], (6,), activation='relu', padding='same', name=name + '_conv2')(up1)
        return conv2

    input_subject = Input(shape=input_shape[0], name='input_subject')
    input_sub_noised = GaussianNoise(stddev=.001)(input_subject)
    input_object = Input(shape=input_shape[1], name='input_object')
    input_rel = Input(shape=input_shape[2], name='input_rel')
    input_rel_noised = GaussianNoise(stddev=.001)(input_rel)

    encode_subject = encode_embedding_input(input_subject)
    encode_rel = encode_embedding_input(input_rel)

    x = concatenate([encode_subject, encode_rel])
    latent = Dense(INNER_SIZE, activation='sigmoid', name='embedding')(x)

    output_object = decode_embedding_input(latent, 'output_object')

    model = Model(inputs=[input_subject, input_object, input_rel],
                  outputs=[input_sub_noised, output_object, input_rel_noised])

    return model
