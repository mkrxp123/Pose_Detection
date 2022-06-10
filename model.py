# %%
from tensorflow import constant, Variable
from tensorflow.keras import Model, Sequential, layers, losses

# %%
def convModel(img = (128, 96, 3), features = 1000):
    model = Sequential([
        layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = img),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation = 'relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation = 'relu'),
        layers.Flatten(),
        layers.Dense(features, activation = 'relu'),
        layers.Dense(features, activation = 'relu', name = "final")
    ])
    return model

def poseModel(img = (160, 90, 3), sequence = 5, features = 1000, num_class = 4):
    conv = convModel(img, features)
    input = layers.Input(shape = (sequence,) + img, name = 'input')
    x = layers.TimeDistributed(conv, name = 'seq_conv')(input)
    x = layers.BatchNormalization()(x)
    key = layers.Dense(x.shape[-1], use_bias = False, name = 'key')(x)
    query = layers.Dense(x.shape[-1], use_bias = False, name = 'query')(x)
    value = layers.Dense(x.shape[-1], use_bias = False, name = 'value')(x)
    x = layers.Attention(name = 'attention')([query, value, key])
    x = layers.Flatten(name = 'flatten')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(100, activation = 'relu')(x)
    x = layers.Dense(num_class, activation = 'softmax', name = 'final')(x)
    model = Model(input, x)
    return model