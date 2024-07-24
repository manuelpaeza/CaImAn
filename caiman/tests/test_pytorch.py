#!/usr/bin/env python

import numpy as np
import os
import keras 

from caiman.paths import caiman_datadir
from caiman.utils.utils import load_graph

try:
    os.environ["KERAS_BACKEND"] = "torch"
    from keras.models import load_model 
    use_keras = True
except(ModuleNotFoundError):
    import torch 
    use_keras = False

def test_torch():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    try:
        model_name = os.path.join(caiman_datadir(), 'model', 'cnn_model')
        if use_keras:
            model_file = model_name + ".h5"

            loaded_model = load_model(model_file)
            loaded_model.load_weights(model_name + '.weights.h5')
            loaded_model.compile('sgd', 'mse')
    except:
        raise Exception(f'NN model could not be loaded. use_keras = {use_keras}')

    A = np.random.randn(10, 50, 50, 1)
    try:
        if use_keras:
            print("hi")
            predictions = loaded_model.predict(A, batch_size=32)
    except:
        raise Exception('NN model could not be deployed. use_keras = ' + str(use_keras))

if __name__ == "__main__":
    test_torch()