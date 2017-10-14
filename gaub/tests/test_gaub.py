from gaub.gaub import Matrix
from fractions import Fraction


def test_add_row():
    m = Matrix.make([[0, 0, 1], [0, 1, 2], [1, 2, 3]])
    m.addRow(1, [- 2 * entry for entry in m[0]])
    assert m == Matrix.make([[0, 0, 1], [0, 1, 0], [1, 2, 3]])


def test_gauss():
    m = Matrix.make([[0, 0], [0, 0]]).gauss()
    assert m == Matrix.make([[0, 0], [0, 0]])

    m = Matrix.make([[1, 2], [1, 2]]).gauss()
    assert m == Matrix.make([[1, 2], [0, 0]])

    m = Matrix.make([[1, 2, 3], [0, 1, 2], [0, 0, 1]]).gauss()
    assert m == Matrix.make([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    m = Matrix.make([[0, 0, 1], [0, 1, 2], [1, 2, 3]]).gauss()
    assert m == Matrix.make([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    m = Matrix.make([[1, 5, 7], [-2, -7, -5]]).gauss()
    assert m == Matrix.make([[1, 0, -8], [0, 1, 3]])

    m = Matrix.make([
        [1, 2, 3, 4],
        [5, 7, 1, 1],
        [1, 1, 11, 2],
        [1, 2, 3, 4]]).gauss()
    assert m == Matrix.make([[1, 0, 0, -13 / 2],
                             [0, 1, 0, 90/19],
                             [0, 0, 1, 13/38],
                             [0, 0, 0, 0]])

    matrix = [list(map(Fraction, row))
              for row in [[1, 2, 3, 4],
                          [5, 7, 1, 1],
                          [1, 1, 11, 2],
                          [1, 2, 3, 4]]]
    m = Matrix.make(matrix).gauss()
    rowEchMatrix = [list(map(Fraction, row))
                    for row in [[1, 0, 0, -13 / 2],
                                [0, 1, 0, 90/19],
                                [0, 0, 1, 13/38],
                                [0, 0, 0, 0]]]
    assert m == Matrix.make(rowEchMatrix)


def test_multiplication():
    m = Matrix.make([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    M = Matrix.make([[1, 2, 3], [0, 1, 2], [0, 0, 1]])
    assert m * M == Matrix.make([[0, 0, 1], [0, 0, 0], [0, 0, 0]])

def test_transposition():
    M = Matrix.make([[1, 2, 3], [0, 1, 2], [0, 0, 1]]).transposed()
    assert M == Matrix.make([[1, 0, 0], [2, 1, 0], [3, 2, 1]])
