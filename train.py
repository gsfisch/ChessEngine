import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import random
from encoding import encoding
from neural_network import get_neural_network
import os
from tensorflow.python.client import device_lib
import sys


# Fixes the evaluation value from the dataset
def fix_eval(eval):
    if '#+' in eval:
        eval = 10_000

    elif '#-' in eval:
        eval = -10_000

    elif '+' in eval:
        eval = int(eval[1:])

    elif '-' in eval:
        eval = -1 * int(eval[1:])

    else:
        eval = int(eval)

    return eval


# Driver code for training the neural network
def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    file_number = 1

    if len(sys.argv) == 1:
        if sys.argv[0] == 0:
            file_number = ""
        else:
            file_number = sys.argv[0]

    print(device_lib.list_local_devices())

    with open(f'Data_set\chessData{file_number}.csv', encoding='utf-8-sig') as data:

        # Reads label row
        data.readline()

        train_positions = []
        train_evaluations = []


        # Extract position and evaluation from Dataset
        '''
        for _ in range(12_000_000):
            line = data.readline()

            fen, eval = line.split(sep=',')

            position = encoding(fen)
            eval = fix_eval(eval)            

            train_positions.append(position)
            train_evaluations.append(eval)
        '''

        for row in data.readlines():
            fen, eval = row.split(sep=',')

            position = encoding(fen)
            eval = fix_eval(eval)            

            train_positions.append(position)
            train_evaluations.append(eval)

        train_positions = np.array(train_positions)
        train_evaluations = np.array(train_evaluations)
        
        model = get_neural_network()
        #model = keras.models.load_model('model.keras')

        model.fit(train_positions, train_evaluations, epochs=1)

        model.save('model.keras')


if __name__ == "__main__":
    main()
