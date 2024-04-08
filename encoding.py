import numpy as np


# One-hot-encoding for the dataset
def encoding(fen):
    #matrix = np.arange(896).reshape(14, 8, 8)
    #matrix = np.zeros((14, 8, 8))
    matrix = np.zeros((8, 8))
    end_of_position = fen.find(' ')
    position = fen[:end_of_position]
    list_of_rows = position.split('/')

    piece_values = {
        'p': -100,
        'r': -500,
        'n': -300,
        'b': -300,
        'q': -900,
        'k': -10000,
        'P': 100,
        'R': 500,
        'N': 300,
        'B': 300,
        'Q': 900,
        'K': 10000
        }

    j = 0
    
    for row in list_of_rows:
        i = 0
        
        for character in row:
            if character.isnumeric():
                i = i + int(character)
            else:
                matrix[j][i] = piece_values[character]
                i = i + 1

        j = j + 1
        

    return matrix

'''
string = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

print(encoding(string))
'''