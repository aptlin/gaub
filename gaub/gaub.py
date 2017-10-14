#!/usr/bin/env python
# -*- coding: utf-8 -*-


# * Libraries


from fractions import Fraction


# * Variables


epsilon = 0.00000001

# spacing before the start of a line with an intermediate step
offset = 4


# * Helpers


def sign(num):
    return - 2 * int(num < 0) + 1


def tailMax(givenList, start):
    maxValue = givenList[start]
    maxCount = start
    count = start
    while count < len(givenList):
        if maxValue < givenList[count]:
            maxValue = givenList[count]
            maxCount = count
        count += 1
    return maxCount, maxValue


def firstNonzero(givenList):
    return next((idx for idx, value in enumerate(givenList)
                 if abs(value) > epsilon),
                0)


def indentedStr(obj, offset=offset):
    return "\n".join(
        map(
            lambda line: " " * offset + line, str(obj).split("\n")
        )
    )


def prefixedStr(prefixObj, obj):
    from itertools import zip_longest
    objStrLines = str(obj).split('\n')
    prefixStrLines = str(prefixObj).split('\n')
    prefixLen = max(map(lambda element: len(str(element)), prefixStrLines))
    lines = []
    for prefixLine, objStrLine in zip_longest(prefixStrLines,
                                              objStrLines,
                                              fillvalue=''):
        lines.append(prefixLine
                     + ' ' * (prefixLen - len(prefixLine))
                     + objStrLine)
    return '\n'.join(lines)


def wholify(line):
    return list(map(int, line.split()))


def fractionify(givenList):
    return list(map(Fraction, givenList))


# * Classes


# ** Matrix Class


class MatrixError(Exception):
    pass


class Matrix:
    def __init__(self, m=0, n=0, init=True):
        if init:
            self.rows = [[0] * n for _ in range(m)]
        else:
            self.rows = []

    @property
    def nrows(self):
        return len(self.rows)

    @property
    def ncols(self):
        if self.nrows == 0:
            return 0
        else:
            return len(self.rows[0])

    def size(self):
        return self.nrows, self.ncols

    def __eq__(self, matrix):
        firstValues = [entry for row in self.rows for entry in row]
        secondValues = [entry for row in matrix.rows for entry in row]
        for firstEntry, secondEntry in zip(firstValues,
                                           secondValues):
            if abs(firstEntry - secondEntry) > epsilon:
                return False
        return True

    def __getitem__(self, row):
        if row >= self.nrows:
            raise MatrixError("No row {} found.".format(row + 1))
        else:
            return self.rows[row]

    def __setitem__(self, row, newRow):
        if row >= self.nrows:
            raise MatrixError("No row {} found.".format(row + 1))
        elif len(newRow) != self.ncols:
            raise MatrixError("The number of columns do not match.")
        else:
            self.rows[row] = newRow

    def __add__(self, matrix):
        if self.size() != matrix.size():
            raise MatrixError("Matrix sizes do not match, aborting.")
        else:
            resultantMatrix = Matrix(self.nrows, self.ncols)
            for rowIndex in range(self.nrows):
                row = [sum(entry) for entry in zip(self.rows[rowIndex],
                                                   matrix.rows[rowIndex])]
                resultantMatrix[rowIndex] = row
            return resultantMatrix

    def __sub__(self, matrix):
        if self.size() != matrix.size():
            raise MatrixError("Matrix sizes do not match, aborting.")
        else:
            resultantMatrix = Matrix(self.nrows, self.ncols)
            for rowIndex in range(self.nrows):
                row = [entry[0] - entry[1] for entry
                       in zip(self.rows[rowIndex], matrix.rows[rowIndex])]
                resultantMatrix[rowIndex] = row
            return resultantMatrix

    def __mul__(self, matrix):
        isAlternative = isinstance(matrix, int) \
                  or isinstance(matrix, float) \
                  or isinstance(matrix, Fraction)
        if isinstance(matrix, Matrix):
            if self.ncols != matrix.nrows:
                raise MatrixError("Matrices do not have correct dimensions " +
                                  "for the multiplication, aborting.")
            else:
                product = Matrix(self.nrows, matrix.ncols)
                tMultiplier = matrix.transposed()
                for row in range(self.nrows):
                    for col in range(tMultiplier.ncols):
                        product[row][col] = sum([entryPair[0] * entryPair[1]
                                                 for entryPair
                                                 in zip(self.rows[row],
                                                        tMultiplier[col])])
                return product
        elif isAlternative:
            product = Matrix(self.nrows, self.ncols)
            for rowIndex in range(self.nrows):
                product[rowIndex] = list(map(lambda x: x * matrix,
                                             self.rows[rowIndex][:]))
            return product
        else:
            raise TypeError("Cannot multiply Matrix object with {}"
                            .format(type(matrix)))

    def __rmul__(self, factor):
        isAlternative = isinstance(factor, int) \
                  or isinstance(factor, float) \
                  or isinstance(factor, Fraction)
        if isAlternative:
            return self * factor
        else:
            raise TypeError("Cannot multiply Matrix object with {}"
                            .format(type(factor)))

    def __iadd__(self, matrix):
        tmpMatrix = self + matrix
        self.rows = tmpMatrix.rows[:]
        return self

    def __isub__(self, matrix):
        tmpMatrix = self - matrix
        self.rows = tmpMatrix.rows[:]
        return self

    def __imul__(self, matrix):
        tmpMatrix = self * matrix
        self.rows = tmpMatrix.rows[:]
        return self

    def __str__(self):
        if not self.nrows or not self.ncols:
            return '\n'
        else:
            columns = self.columns()
            maxColLens = [max(map(lambda element: len(str(element)), col))
                          for col in columns]
            strColumns = [
                list(
                    map(
                        lambda entry: " " * (colLen - len(str(entry)))
                        + str(entry),
                        column
                    )
                )
                for column, colLen in zip(columns, maxColLens)
            ]
            middles = [" ".join(middle) for middle in zip(*strColumns)]
            strRepresentation = ''
            for rowIndex in range(self.nrows):
                middle = middles[rowIndex]
                if rowIndex == self.nrows - 1:
                    strRepresentation += '|_ ' + middle + ' _|'
                else:
                    strRepresentation += '|  ' + middle + '  |\n'
            strRepresentation = '\n _ ' + ' ' * len(middle) + ' _\n'\
                                + strRepresentation
            return strRepresentation

    def __repr__(self):
        return "Matrix: {}, Size: {}".format(self.rows, self.size())

    @classmethod
    def _makeMatrix(cls, rows):
        nrows = len(rows)
        if nrows == 0:
            return cls()
        else:
            ncols = len(rows[0])
            if ncols == 0:
                return cls()
            elif any([len(row) != ncols for row in rows[1:]]):
                raise MatrixError("Inconsistent number of columns passed.")
            else:
                matrix = cls(nrows, ncols, init=False)
                matrix.rows = rows
            return matrix

    @classmethod
    def make(cls, listOfRows):
        return cls._makeMatrix(listOfRows[:])

    @classmethod
    def cmake(cls, listOfCols):
        matrix = cls._makeMatrix(listOfCols[:])
        matrix.transpose()
        return matrix

    @classmethod
    def identity(cls, nrows, ncols):
        matrix = cls(nrows, ncols)
        for idx in range(min(nrows, ncols)):
            matrix[idx][idx] = 1
        return matrix

    def copy(self):
        return self._makeMatrix(self.rows[:])

    def transpose(self):
        self.rows = [list(column) for column in zip(*self.rows)]

    def transposed(self):
        transposedMatrix = Matrix(self.ncols, self.nrows)
        transposedMatrix.rows = [list(column) for column in zip(*self.rows)]
        return transposedMatrix

    def column(self, index):
        if index >= self.ncols:
            raise MatrixError("No column {} found.".format(index + 1))
        else:
            return [row[index] for row in self.rows]

    def columns(self):
        cols = []
        for colIndex in range(self.ncols):
            cols.append(self.column(colIndex))
        return cols

    def augment(self, columns):
        matrix = Matrix.cmake(columns)
        if matrix.nrows != self.nrows:
            raise MatrixError("Sizes do not match, aborting.")
        else:
            for rowIndex in range(matrix.nrows):
                self.rows[rowIndex] += matrix.rows[rowIndex]

    def augmented(self, columns):
        augmentedMatrix = self.copy()
        augmentedMatrix.augment(columns)
        return augmentedMatrix

    def scaleRow(self, row, factor):
        if self.nrows <= row:
            raise MatrixError("No row {} found, aborting.".format(row + 1))
        else:
            self[row] = [factor * entry for entry in self[row]]

    def addRow(self, row, addedRow):
        if self.nrows <= row:
            raise MatrixError("No row {} found, aborting.".format(row + 1))
        elif len(addedRow) != self.ncols:
            raise MatrixError("Column sizes do not match, aborting."
                              .format(row + 1))
        else:
            self[row] = [x[0] + x[1] for x in zip(self[row], addedRow)]

    def swapRows(self, firstRow, secondRow, verbose=False):
        if firstRow != secondRow:
            if verbose:
                print(indentedStr("Swapped row {} and {}."
                                  .format(firstRow, secondRow)))
            tmpRow = self[secondRow]
            self[secondRow] = self[firstRow]
            self[firstRow] = tmpRow
            if verbose:
                print(indentedStr(self))

    def _findClosestPositiveColumnMax(self, colIndex):
        minSize = min(self.ncols, self.nrows)
        absColumn = list(map(abs, self.column(colIndex)))
        maxRowIndex, maxValue = tailMax(absColumn, colIndex)
        maxColIndex = colIndex
        if maxValue <= epsilon:
            nextIndex = colIndex + 1
            while nextIndex < minSize - 1:
                nextColumn = list(map(abs, self.column(nextIndex)))
                rowIndex, element = tailMax(nextColumn, colIndex)
                if abs(element) > epsilon:
                    maxRowIndex, maxValue = rowIndex, element
                    maxColIndex = nextIndex
                    break
                nextIndex += 1
        return maxRowIndex, maxColIndex, maxValue

    def gauss(self, verbose=False):
        matrix = self
        if not matrix.rows:
            return matrix
        else:
            if verbose:
                print("Initial state:")
                print(indentedStr(matrix))
                # initialise the count used to number steps
                count = 1
            for colIndex in range(min(matrix.ncols, matrix.nrows)):
                if verbose:
                    print(("\n{}. Checking the order "
                           + "of elements in column {}...")
                          .format(count, colIndex + 1))
                    count += 1
                maxRowIndex, \
                    maxColIndex, \
                    maxValue = matrix._findClosestPositiveColumnMax(colIndex)
                matrix.swapRows(colIndex, maxRowIndex, verbose)

                if abs(maxValue) > epsilon:
                    if verbose:
                        print("\n{}. Pivoting row {}..."
                              .format(count, colIndex + 1))
                        count += 1
                    factor = sign(matrix[colIndex][maxColIndex]) / maxValue
                    if verbose:
                        print(indentedStr("Scaled row {} by the factor of {}:"
                                          .format(maxColIndex, factor)))
                    matrix.scaleRow(colIndex, factor)
                    if verbose:
                        print(indentedStr(matrix))

                maxRow = matrix[colIndex]
                nonzeroIdx = maxColIndex + firstNonzero(maxRow[maxColIndex:])

                if verbose and colIndex + 1 < matrix.nrows:
                    print("\n{}. Conducting forward elimination:"
                          .format(count))
                    forwardCount = 1

                for rowIndex in range(colIndex + 1, matrix.nrows):
                    row = matrix[rowIndex]
                    subtrahend = [- row[nonzeroIdx] * entry
                                  for entry in maxRow]
                    if subtrahend != [0] * len(subtrahend):
                        matrix.addRow(rowIndex, subtrahend)
                        if verbose:
                            print(
                                indentedStr(
                                    "{}.{}. Reduced row {} using row {}:"
                                    .format(count, forwardCount,
                                            colIndex + 1, rowIndex + 1)
                                )
                            )
                            print(indentedStr(matrix))
                            forwardCount += 1
                if verbose and colIndex + 1 < matrix.nrows:
                    count += 1

                if verbose and colIndex - 1 > -1:
                    print("\n{}. Conducting backward elimination:"
                          .format(count))
                    backwardCount = 1
                for rowIndex in range(colIndex - 1, -1, -1):
                    row = matrix[rowIndex]
                    subtrahend = [- row[nonzeroIdx] * entry
                                  for entry in maxRow]
                    if subtrahend != [0] * len(subtrahend):
                        matrix.addRow(rowIndex, subtrahend)
                        if verbose:
                            print(
                                indentedStr(
                                    "{}.{}. Reduced row {} using row {}:"
                                    .format(count, backwardCount,
                                            colIndex + 1, rowIndex + 1)
                                )
                            )
                            print(indentedStr(matrix))
                            backwardCount += 1
                    if verbose and colIndex - 1 > -1:
                        count += 1
            if verbose:
                print("-" * 40)
                print("Reduced matrix:")
            return matrix


# ** Augmented Matrix Class


class AugmentedMatrix(Matrix):
    def __init__(self, m=0, n=0, offset=1, init=True):
        super().__init__(m, n)
        self.offset = offset

    def __str__(self):
        if not self.nrows or not self.ncols:
            return '\n'
        else:
            columns = self.columns()
            maxColLens = [max(map(lambda element: len(str(element)), col))
                          for col in columns]
            strColumns = [
                list(
                    map(
                        lambda entry: " " * (colLen - len(str(entry)))
                        + str(entry),
                        column
                    )
                )
                for column, colLen in zip(columns, maxColLens)
            ]
            middles = [' '.join(middle)
                       for middle in zip(
                               *(strColumns[:-(self.offset)]
                                 + [["|"] * self.nrows]
                                 + strColumns[-(self.offset):]))]
            strRepresentation = ''
            for rowIndex in range(self.nrows):
                middle = middles[rowIndex]
                if rowIndex == self.nrows - 1:
                    strRepresentation += '|_ ' + middle + ' _|' + '\n'
                else:
                    strRepresentation += '|  ' + middle + '  |\n'
            strRepresentation = '\n _ ' + ' ' * len(middle) + ' _\n'\
                                + strRepresentation
            return strRepresentation
