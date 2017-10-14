# * Libraries


from gaub import fractionify
from gaub import AugmentedMatrix


# * Code


def solve(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        side = int(f.readline())
        matrix = AugmentedMatrix.make(
            [fractionify(f.readline().split()) for _ in range(side)]
        )
        vector = fractionify(f.readline().split())
        augMatrix = matrix.augmented([vector])
        reducedAugMatrix = augMatrix.gauss(verbose=True)
        return reducedAugMatrix
