# * Libraries


from gaub import Matrix, AugmentedMatrix
from gaub import wholify, fractionify


# * Code

# representation invariant: input matrices are upper-triangular
with open("example_4_input.txt", 'r', encoding='utf-8') as f:
    count, nrows = wholify(f.readline())
    matrices = []
    while count > 0:
        matrix = [fractionify(f.readline().split()) for _ in range(nrows)]
        matrices.append(AugmentedMatrix.make(matrix))
        count -= 1
    for matrix in matrices:
        print("Given matrix:", matrix)
        matrix.offset = matrix.ncols
        matrix.augment(Matrix(matrix.nrows, matrix.ncols).columns())
        eigenvalues = set(matrix[idx][idx] for idx in range(min(matrix.nrows,
                                                                matrix.ncols)))
        for value in sorted(eigenvalues):
            message = "Reducing the matrix with respect to the eigenvalue {}:"\
                      .format(value)
            print("\n{}\n{}\n{}\n".format(len(message) * "#",
                                          message,
                                          len(message) * "#")
                  )
            reducedMatrix = AugmentedMatrix(matrix.nrows, matrix.ncols,
                                            offset=matrix.offset, init=False)
            reducedMatrix.rows = (matrix
                                  - value * Matrix.identity(matrix.nrows,
                                                            matrix.ncols)).rows
            print(reducedMatrix.gauss(verbose=True))
