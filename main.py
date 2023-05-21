import cv2
import imutils
from tensorflow.keras.models import load_model
import numpy as np
from Sudoku import solveSudoku
from tensorflow.keras.preprocessing.image import img_to_array
from utils import find_puzzle, extract_digit, display_numbers_on_board

# Sudoku Solver
model = load_model('model/model_mnist/')
img_path = 'sudoku images/6.png'
img_shape = [28,28]

img = cv2.imread(img_path)
img = imutils.resize(img, width=600)
puzzle, warped = find_puzzle(img)
puzzle = imutils.resize(puzzle, width=600)
warped = imutils.resize(warped, width=600)
step_x = warped.shape[1] // 9
step_y = warped.shape[0] // 9
board = np.zeros(shape=(9, 9), dtype='int')
cell_locs = []
for i in range(9):
    for j in range(9):
        topleftx = j * step_x
        toplefty = i * step_y
        rightendx = (j + 1) * step_x
        rightendy = (i + 1) * step_y
        cell = warped[toplefty:rightendy, topleftx:rightendx]
        digit = extract_digit(cell)
        if digit is not None:
            roi = cv2.resize(digit, tuple(img_shape))
            roi = roi.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            pred = model.predict(roi).argmax(axis=1)[0]
            board[i, j] = pred
        cell_locs.append([topleftx, toplefty, rightendx, rightendy])

_ = display_numbers_on_board(board, puzzle, cell_locs)
while 1:
    res = input('Check all given numbers are predicted correctly? (y/n)')
    if res == 'n':
        cx, cy, ele = input('Input row no, col no, the correct number of cell (counting starts from 0) For eg. --> 1 2 1 :  ').split()
        try:
            board[int(cx), int(cy)] = int(ele)
        except:
            print('out of range...')
        _ = display_numbers_on_board(board, puzzle, cell_locs)
    elif res == 'y':
        break
    else:
        print('Wrong choice!!!')

solved = solveSudoku(board)
x = display_numbers_on_board(board, puzzle, cell_locs)
cv2.imshow('Sudoku is solved!', x)
cv2.waitKey(0)
cv2.destroyAllWindows()
