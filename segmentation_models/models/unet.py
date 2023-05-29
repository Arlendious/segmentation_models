from keras_applications import get_submodules_from_kwargs
from tensorflow.keras import layers, models
from ._common_blocks import SeparableConv2dBnReLU
from ._utils import freeze_model, filter_keras_submodules
from ..backbones.backbones_factory import Backbones

backend = None
layers = None
models = None
keras_utils = None


# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def attention_block(query, key, value, relative_position_bias=None, dropout_rate=0.0):
    q_shape = tf.shape(query)
    k_shape = tf.shape(key)
    v_shape = tf.shape(value)
    
    # Ensure compatible shapes for matrix multiplication
    assert q_shape[-1] == k_shape[-1], "Last dimension of query and key must be the same"
    assert k_shape[1] == v_shape[1], "Second dimension of key and value must be the same"
    assert k_shape[2] == v_shape[2], "Third dimension of key and value must be the same"
    
    query = tf.reshape(query, [q_shape[0], q_shape[1] * q_shape[2], q_shape[3]])
    key = tf.reshape(key, [k_shape[0], k_shape[1] * k_shape[2], k_shape[3]])
    value = tf.reshape(value, [v_shape[0], v_shape[1] * v_shape[2], v_shape[3]])
    
    score = tf.matmul(query, key, transpose_b=True)
    n_key = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_score = score / tf.math.sqrt(n_key)
    
    if relative_position_bias is not None:
        scaled_score = scaled_score + relative_position_bias
    
    weight = tf.nn.softmax(scaled_score, axis=-1)
    
    if dropout_rate > 0.0:
        weight = tf.nn.dropout(weight, rate=dropout_rate)
    
    out = tf.matmul(weight, value)
    
    return out

def SeparableConv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return SeparableConv2dBnReLU(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False, attention=False, dropout_rate=0.0):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            if attention:
                x = attention_block(x, skip, value=x)
            else:
                x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = SeparableConv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = SeparableConv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False, attention=False, dropout_rate=0.0):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor, skip=None):
        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            if attention:
                x = attention_block(x, skip, value=x)
            else:
                x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = SeparableConv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer


# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_unet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
        attention=False,
        dropout_rate=0.0,
):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = SeparableConv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = SeparableConv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        if attention: 
            x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm, attention=attention,
                              dropout_rate=dropout_rate)(x, skip)
        else:
            x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model


# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------

def Unet(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        attention=False,
        dropout_rate=0.0,
        **kwargs
):
    """ Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
            case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
            # able to process images at any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for the last model layer
            (e.g., ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of the encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from the top of the model.
            Each of these layers will be concatenated with the corresponding decoder block. If ``default`` is used
            layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_block_type: one of the blocks with the following layers structure:

            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D``

        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalization`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        attention: if ``True``, attach an attention block to the decoder blocks.
        dropout_rate: dropout rate used in the decoder blocks if attention is enabled.

    Returns:
        ``keras.models.Model``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)

    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_unet(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
        attention=attention,
        dropout_rate=dropout_rate,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
