def solveSudoku(board):
    for i in range(9):
        for j in range(9):
            board[i][j] = int(board[i][j]) if board[i][j] != '.' else 0

    def return_empty(board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return i, j
        return None

    def check(board, num, pos):
        row, col = pos

        # checking row
        for i in range(9):
            if board[row][i] == num and i != col:
                return False

        # checking col
        for i in range(9):
            if board[i][col] == num and i != row:
                return False

        # checking for sub-grid
        box_x = row // 3
        box_y = col // 3

        for i in range(box_x * 3, box_x * 3 + 3):
            for j in range(box_y * 3, box_y * 3 + 3):
                if board[i][j] == num and (i, j) != pos:
                    return False
        return True

    def solve(board):
        vacant = return_empty(board)
        if vacant is None: return True  # means all cells are filled now and we have completed the sudoku
        row, col = vacant
        for i in range(1, 10):
            if check(board, i, vacant):
                board[row][col] = i
                if solve(board):
                    return True
                board[row][col] = 0
        return False

    solve(board)

    return board

# board = [[8, 0, 0, 0, 1, 0, 0, 0, 9],
#        [0, 5, 0, 8, 0, 7, 0, 1, 0],
#        [0, 0, 4, 0, 9, 0, 7, 0, 0],
#        [0, 6, 0, 7, 0, 1, 0, 2, 0],
#        [5, 0, 8, 0, 6, 0, 1, 0, 7],
#        [0, 1, 0, 5, 0, 2, 0, 9, 0],
#        [0, 0, 7, 0, 4, 0, 6, 0, 0],
#        [0, 8, 0, 3, 0, 9, 0, 4, 0],
#        [3, 0, 0, 0, 5, 0, 0, 0, 8]]

# print(solveSudoku(board))
