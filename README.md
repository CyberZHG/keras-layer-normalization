# Keras Layer Normalization

[![Travis](https://travis-ci.org/CyberZHG/keras-layer-normalization.svg)](https://travis-ci.org/CyberZHG/keras-layer-normalization)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-layer-normalization/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-layer-normalization)
[![Version](https://img.shields.io/pypi/v/keras-layer-normalization.svg)](https://pypi.org/project/keras-layer-normalization/)
![Downloads](https://img.shields.io/pypi/dm/keras-layer-normalization.svg)
![License](https://img.shields.io/pypi/l/keras-layer-normalization.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)


Implementation of the paper: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

## Install

```bash
pip install keras-layer-normalization
```

## Usage

```python
import keras
from keras_layer_normalization import LayerNormalization


input_layer = keras.layers.Input(shape=(2, 3))
norm_layer = LayerNormalization()(input_layer)
model = keras.models.Model(inputs=input_layer, outputs=norm_layer)
model.compile(optimizer='adam', loss='mse', metrics={},)
model.summary()
```
