# * Esfandiar-Kiani *
from Module.Singleton import Singleton


class SudokuSolver(Singleton):
    def __init__(self, _puzzle: list = None) -> None:
        self._puzzle = _puzzle
        self._valid_Numbers = list(range(1, 10))
        if self._puzzle:
            self._row_Count = len(self._puzzle)
            self._column_Count = len(self._puzzle[0])
        else:
            self._row_Count = 0
            self._column_Count = 0
        return

    def _next_Empty(self):
        for i in range(self._row_Count):
            for j in range(self._column_Count):
                if self._puzzle[i][j] <= 0:
                    return i, j
        return None, None

    def _check_Solvable(self):
        for i in range(self._row_Count):
            for j in range(self._column_Count):
                if self._puzzle[i][j] != 0:
                    temp = self._puzzle[i][j]
                    self._puzzle[i][j] = 0
                    
                    if not self._check_Valid(i, j, temp):
                        self.solvable = False
                        return False
                    
                    self._puzzle[i][j] = temp
        return True


    def _check_Valid(self, I: int, J: int, guessNum: int):
        row = self._puzzle[I]
        column = [self._puzzle[i][J] for i in range(self._row_Count)]
        square = []

        square_x = (I // 3) * 3
        square_y = (J // 3) * 3

        for i in range(square_x, square_x + 3):
            for j in range(square_y, square_y + 3):
                square.append(self._puzzle[i][j])

        neighbors = row + column + square
        if guessNum in neighbors:
            return False
        return True

    def _start_Engine(self):
        row, col = self._next_Empty()

        # * Last Return
        if row is None:
            return True

        for guess in self._valid_Numbers:
            if self._check_Valid(row, col, guess):
                self._puzzle[row][col] = guess

                # * Recursive Algorithm
                if SudokuSolver(self._puzzle)._start_Engine():
                    return True

            self._puzzle[row][col] = 0

        # * Return False If Puzzle Is Unsolvable
        return False

    def writeTo_File(self, puzzle, fileName: str, filePath: str):
        if puzzle:
            self.__init__(puzzle)
            
        with open(f'{filePath}{fileName}.txt', 'a+') as file:
            file.write("\n")

            for i in range(self._row_Count):
                if i % 3 == 0:
                    file.write("\n")

                for j in range(self._column_Count):
                    if j % 3 == 0 and j != 0:
                        file.write("\t")
                    file.write(str(self._puzzle[i][j]))
                file.write("\n")

            file.write("\n")
            file.close()
        return

    def print(self, puzzle: list = None):
        if puzzle:
            self.__init__(puzzle)

        for i in range(self._row_Count):
            if i % 3 == 0:
                print()

            for j in range(self._column_Count):
                if j % 3 == 0 and j != 0:
                    print("\t", end="")
                print(self._puzzle[i][j], end="")

            print()

    def solve(self, puzzle: list = None, writeToFile: bool = False, fileName: str = "Solved Sudoku", filePath: str = "./"):
        if puzzle:
            self.__init__(puzzle)


        if self._check_Solvable():
            self._start_Engine()
            if writeToFile:
                self.writeTo_File(self._puzzle, fileName, filePath)
            print(
                f"Puzzle Solved And Writhed To The {filePath}{fileName}.txt .\n See It: ")
            # * Print Solved Puzzle
            self.print()
            return True, self._puzzle
        else:
            print("Sorry...\nThis Puzzle Is Unsolvable!")
            return False, self._puzzle
