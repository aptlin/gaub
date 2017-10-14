# * Libraries


from gaub import Matrix
from gaub import wholify
from gaub import prefixedStr, indentedStr


# * Code


with open("example_3_input.txt", 'r', encoding='utf-8') as f:
    count, nrows = wholify(f.readline())
    matrices = []
    while count > 0:
        matrix = [wholify(f.readline()) for _ in range(nrows)]
        matrices.append(Matrix.make(matrix))
        count -= 1
    assert len(matrices) == 4, \
        "Unexpected number of matrices({}) passed.".format(len(matrices))
    print("Matrices:",
          *[prefixedStr(prefix, indentedStr(matrix, 2))
            for prefix, matrix in zip(
                    ["\nA:", "\nB:", "\nC:", "\nD:"],
                    matrices)],
          sep="\n")
    A, B, C, D = matrices
    print("\nTransformation:\n\n",
          "A * B * C",
          "- (C * A^t)^t * C",
          "+ A * D * C",
          "- B^t * A^t * A",
          "+ A * (A^t * C)^t",
          "- D * A^t * A\n")
    print(
        "Result:",
        A*B*C - (C * A.transposed()).transposed() * C + A * D * C
        - B.transposed() * A.transposed() * A
        + A * (A.transposed() * C).transposed() - D * A.transposed() * A
    )
