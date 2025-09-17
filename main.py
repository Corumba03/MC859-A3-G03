from GRASP import Solution
from GRASP.metaheuristics import AbstractGRASP, ReactiveGRASP
from GRASP.problems import Evaluator, QBF, SC_QBF, SetCover


def main():
    
    # Read the number of variables (n)
    n = int(input())

    # Read the number of elements in each set (but not used afterwards)
    _ = list(map(int, input().split()))

    # Initialize list to hold the sets
    sets = []

    # Read n sets of integers, one per line, and convert to 0-based indices
    for _ in range(n):
        sets.append(set(x - 1 for x in map(int, input().split())))


    # Initialize and read the upper triangular matrix
    upper_A = []
    for i in range(n):
        row = list(map(float, input().split()))
        upper_A.append(row)

    # Convert upper triangular to full n x n matrix (symmetric, fill lower triangle)
    A = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(len(upper_A[i])):
            A[i][j] = upper_A[i][j]
            if i != j:
                A[j][i] = upper_A[i][j]

    # Model creation

    solver = ReactiveGRASP(
        obj_function = SC_QBF(n, A, sets),
        iterations=10,
        alpha_pool= [0.8],
        constructive_type='cost_ratio',
        search_type='first'
    )

    solutions = solver.solve()

    for i, sol in enumerate(solutions):
        if i == len(solutions) - 1:
            print(f"\nFinal {sol}")
        else:
            print(f"(Iter. {i+1}) BestSol = {sol}")
    


if __name__=='__main__':
    main()
