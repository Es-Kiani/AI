# README

A Command-Line Sudoku Puzzle Solver Program.

## Puzzle Schematic
```

    0 0 0  |  0 0 0  |  0 0 0
    0 0 0  |  0 0 0  |  0 0 0
    0 0 0  |  0 0 0  |  0 0 0
    ------   -------   -------                               
    0 0 0  |  0 0 0  |  0 0 0
    0 0 0  |  0 0 0  |  0 0 0
    0 0 0  |  0 0 0  |  0 0 0
    ------   -------   -------                               
    0 0 0  |  0 0 0  |  0 0 0
    0 0 0  |  0 0 0  |  0 0 0
    0 0 0  |  0 0 0  |  0 0 0
    
```




## A Puzzle For Solve
```
  puzzle = [
              [0, 0, 0,     1, 0, 2,     0, 3, 7],
              [0, 0, 0,     4, 0, 9,     5, 8, 6],
              [0, 0, 6,     0, 0, 8,     9, 0, 0],

              [0, 7, 0,     0, 0, 0,     8, 6, 5],
              [0, 0, 1,     6, 0, 0,     0, 0, 3],
              [6, 3, 0,     2, 0, 0,     0, 0, 1],

              [8, 1, 7,     3, 4, 5,     6, 2, 9],
              [3, 9, 0,     0, 2, 0,     0, 5, 0],
              [0, 6, 5,     0, 0, 0,     0, 0, 0]
           ]
           
```

## Usage

How to define the puzzle in correct format?

Use 'SudokuSolver' class  to creat a sudoku solver object and than 'solve' method to solve puzzle and print it to console.
Also you can use 'print' method to print any puzzle that you define it befor in the code.

If you want to save your solved puzzle into a '*.txt' file, you can add "writeToFile=True" argument when call the 'solve' method.

The defult arguments of 'solve' method for writing to a '*.txt' file is: 
```
def solve(self, puzzle: list = None, writeToFile: bool = False, fileName: str = "Solved Sudoku", filePath: str = "./"): -> Bool , puzzle ...
```


## Code

```
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


SudokuSolver().solve(puzzle)
# SudokuSolver().solve(puzzle, writeToFile=True)
# SudokuSolver().solve(puzzle, True, fileName = "Solved Sudoku")
# SudokuSolver().solve(puzzle, True, "Solved Sudoku", filePath = "./")

# SudokuSolver().print(puzzle)



# solver = SudokuSolver()
# solver.solve(puzzle)
# solver.solve(puzzle, writeToFile=True)
# solver.solve(puzzle, True, fileName = "Solved Sudoku")
# solver.solve(puzzle, True, "Solved Sudoku", filePath = "./")

# solver.print(puzzle)

```

## Copyrights

.:: Made By Me [-Esfandiar Kiani] ::.
