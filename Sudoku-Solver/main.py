from Module.SudokuSolver import SudokuSolver

puzzle = [
    [0, 0, 0, 1, 0, 2, 0, 3, 7],
    [0, 0, 0, 4, 0, 9, 5, 8, 6],
    [0, 0, 6, 0, 0, 8, 9, 0, 0],
    [0, 7, 0, 0, 0, 0, 8, 6, 5],
    [0, 0, 1, 6, 0, 0, 0, 0, 3],
    [6, 3, 0, 2, 0, 0, 0, 0, 1],
    [8, 1, 7, 3, 4, 5, 6, 2, 9],
    [3, 9, 0, 0, 2, 0, 0, 5, 0],
    [0, 6, 5, 0, 0, 0, 0, 0, 0]
]

if __name__=="__main__":
    # SudokuSolver().print(puzzle)
    is_Solve, solved_Puzzle = SudokuSolver().solve(puzzle, writeToFile=True)


