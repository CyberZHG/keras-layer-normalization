import os
import tempfile
import random
import unittest
import keras
import numpy as np
from keras_multi_head import MultiHeadAttention
from keras_layer_normalization import LayerNormalization


class TestLayerNormalization(unittest.TestCase):

    def test_sample(self):
        input_layer = keras.layers.Input(
            shape=(2, 3),
            name='Input',
        )
        norm_layer = LayerNormalization(
            name='Layer-Normalization',
        )(input_layer)
        model = keras.models.Model(
            inputs=input_layer,
            outputs=norm_layer,
        )
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics={},
        )
        model.summary()
        inputs = np.array([[
            [0.2, 0.1, 0.3],
            [0.5, 0.1, 0.1],
        ]])
        predict = model.predict(inputs)
        expected = np.asarray([[
            [0.0, -1.22474487, 1.22474487],
            [1.41421356, -0.707106781, -0.707106781],
        ]])
        self.assertTrue(np.allclose(expected, predict), predict)

        input_layer = keras.layers.Input(
            shape=(10, 256),
            name='Input',
        )
        norm_layer = LayerNormalization(
            name='Layer-Normalization',
            beta_initializer='ones',
        )(input_layer)
        model = keras.models.Model(
            inputs=input_layer,
            outputs=norm_layer,
        )
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics={},
        )
        model.summary()
        inputs = np.zeros((2, 10, 256))
        predict = model.predict(inputs)
        expected = np.ones((2, 10, 256))
        self.assertTrue(np.allclose(expected, predict))

    def test_fit(self):
        def _leaky_relu(x):
            return keras.activations.relu(x, alpha=0.01)

        input_layer = keras.layers.Input(
            shape=(2, 3),
            name='Input',
        )
        norm_layer = LayerNormalization(
            name='Layer-Normalization-1',
            trainable=False,
        )(input_layer)
        att_layer = MultiHeadAttention(
            head_num=3,
            activation=_leaky_relu,
            name='Multi-Head-Attentions'
        )(norm_layer)
        dense_layer = keras.layers.Dense(units=3, name='Dense-1')(att_layer)
        norm_layer = LayerNormalization(
            name='Layer-Normalization-2',
            trainable=False,
        )(dense_layer)
        dense_layer = keras.layers.Dense(units=3, name='Dense-2')(norm_layer)
        model = keras.models.Model(
            inputs=input_layer,
            outputs=dense_layer,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss='mse',
            metrics={},
        )
        model.summary()

        def _generator(batch_size=32):
            while True:
                batch_inputs = np.random.random((batch_size, 2, 3))
                batch_outputs = np.asarray([[[0.0, -0.1, 0.2]] * 2] * batch_size)
                yield batch_inputs, batch_outputs

        model.fit_generator(
            generator=_generator(),
            steps_per_epoch=1000,
            epochs=10,
            validation_data=_generator(),
            validation_steps=100,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            ],
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_layer_normalization_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            '_leaky_relu': _leaky_relu,
            'MultiHeadAttention': MultiHeadAttention,
            'LayerNormalization': LayerNormalization,
        })
        for inputs, _ in _generator(batch_size=3):
            predicts = model.predict(inputs)
            expect = np.round(np.asarray([[[0.0, -0.1, 0.2]] * 2] * 3), decimals=1)
            actual = np.round(predicts, decimals=1)
            self.assertTrue(np.allclose(expect, actual), (expect, actual))
            break

    def test_fit_zeros(self):
        def _leaky_relu(x):
            return keras.activations.relu(x, alpha=0.01)

        input_layer = keras.layers.Input(
            shape=(2, 3),
            name='Input',
        )
        norm_layer = LayerNormalization(
            name='Layer-Normalization-1',
            trainable=False,
        )(input_layer)
        att_layer = MultiHeadAttention(
            head_num=3,
            activation=_leaky_relu,
            name='Multi-Head-Attentions'
        )(norm_layer)
        dense_layer = keras.layers.Dense(units=3, name='Dense-1')(att_layer)
        norm_layer = LayerNormalization(
            name='Layer-Normalization-2',
            trainable=False,
        )(dense_layer)
        dense_layer = keras.layers.Dense(units=3, name='Dense-2')(norm_layer)
        model = keras.models.Model(
            inputs=input_layer,
            outputs=dense_layer,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss='mse',
            metrics={},
        )
        model.summary()

        def _generator_zeros(batch_size=32):
            while True:
                batch_inputs = np.zeros((batch_size, 2, 3))
                batch_outputs = np.asarray([[[0.0, -0.1, 0.2]] * 2] * batch_size)
                yield batch_inputs, batch_outputs

        model.fit_generator(
            generator=_generator_zeros(),
            steps_per_epoch=1000,
            epochs=10,
            validation_data=_generator_zeros(),
            validation_steps=100,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            ],
        )
        for inputs, _ in _generator_zeros(batch_size=3):
            predicts = model.predict(inputs)
            expect = np.round(np.asarray([[[0.0, -0.1, 0.2]] * 2] * 3), decimals=1)
            actual = np.round(predicts, decimals=1)
            self.assertTrue(np.allclose(expect, actual), (expect, actual))
            break

    def test_save_load_json(self):
        model = keras.models.Sequential()
        model.add(LayerNormalization())
        model.build(input_shape=(2, 3))
        encoded = model.to_json()
        model = keras.models.model_from_json(encoded, custom_objects={'LayerNormalization': LayerNormalization})
        model.summary()
