# Keras Layer Normalization

[![Version](https://img.shields.io/pypi/v/keras-layer-normalization.svg)](https://pypi.org/project/keras-layer-normalization/)
![License](https://img.shields.io/pypi/l/keras-layer-normalization.svg)

Implementation of the paper: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

## Install

```bash
pip install keras-layer-normalization
```

## Usage

```python
from tensorflow import keras
from keras_layer_normalization import LayerNormalization


input_layer = keras.layers.Input(shape=(2, 3))
norm_layer = LayerNormalization()(input_layer)
model = keras.models.Model(inputs=input_layer, outputs=norm_layer)
model.compile(optimizer='adam', loss='mse', metrics={},)
model.summary()
```
