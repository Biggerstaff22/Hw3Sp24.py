import math


def is_symmetric(matrix):
    """
    Check if a matrix is symmetric.

    Args:
        matrix (list of lists): The matrix to check.

    Returns:
        bool: True if the matrix is symmetric, False otherwise.
    """
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True


def is_positive_definite(matrix):
    """
    Check if a matrix is positive definite.

    Args:
        matrix (list of lists): The matrix to check.

    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    # We can use Cholesky decomposition to check positive definiteness
    try:
        _ = cholesky_decomposition(matrix)
        return True
    except Exception:
        return False


def cholesky_decomposition(matrix):
    """
    Perform Cholesky decomposition on a symmetric positive definite matrix.

    Args:
        matrix (list of lists): The matrix to decompose.

    Returns:
        list of lists: Lower triangular matrix L such that L*L^T = matrix.
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                sum_val = sum(L[i][k] ** 2 for k in range(j))
                L[i][j] = math.sqrt(matrix[i][i] - sum_val)
            else:
                sum_val = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = (matrix[i][j] - sum_val) / L[j][j]

    return L


def forward_substitution(L, b):
    """
    Perform forward substitution to solve L*y = b.

    Args:
        L (list of lists): Lower triangular matrix.
        b (list): Vector.

    Returns:
        list: Solution vector.
    """
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y


def backward_substitution(L_transpose, y):
    """
    Perform backward substitution to solve L_transpose*x = y.

    Args:
        L_transpose (list of lists): Transpose of a lower triangular matrix.
        y (list): Vector.

    Returns:
        list: Solution vector.
    """
    n = len(L_transpose)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L_transpose[i][j] * x[j] for j in range(i + 1, n))) / L_transpose[i][i]
    return x


def doolittle_LU_factorization(matrix):
    """
    Perform Doolittle LU factorization.

    Args:
        matrix (list of lists): The matrix to factorize.

    Returns:
        tuple: Tuple containing lower triangular matrix L and upper triangular matrix U.
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0

    for i in range(n):
        for j in range(i, n):
            U[i][j] = matrix[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i + 1, n):
            L[j][i] = (matrix[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U


def solve_linear_equations_cholesky(matrix_A, matrix_B):
    """
    Solve the system of linear equations using Cholesky method.

    Args:
        matrix_A (list of lists): Coefficient matrix.
        matrix_B (list of lists): Right-hand side matrix.

    Returns:
        list: Solution vector.
    """
    if not is_symmetric(matrix_A) or not is_positive_definite(matrix_A):
        raise ValueError("Matrix A is not symmetric positive definite.")

    L = cholesky_decomposition(matrix_A)
    L_transpose = list(map(list, zip(*L)))

    y = forward_substitution(L, matrix_B)
    x = backward_substitution(L_transpose, y)

    return x


def solve_linear_equations_doolittle(matrix_A, matrix_B):
    """
    Solve the system of linear equations using Doolittle method.

    Args:
        matrix_A (list of lists): Coefficient matrix.
        matrix_B (list of lists): Right-hand side matrix.

    Returns:
        list: Solution vector.
    """
    L, U = doolittle_LU_factorization(matrix_A)

    # Forward substitution to solve L*y = B
    y = forward_substitution(L, matrix_B)

    # Backward substitution to solve U*x = y
    x = backward_substitution(U, y)

    return x


# Matrix A and B
A = [
    [1, -1, 3, 2],
    [-1, 5, -5, -2],
    [3, -5, 19, 3],
    [2, -2, 3, 21]
]

B = [
    [4, 2, 4, 0],
    [2, 2, 3, 2],
    [4, 3, 6, 3],
    [0, 2, 3, 9]
]

try:
    # Solve using Cholesky method
    solution_cholesky = solve_linear_equations_cholesky(A, B)
    print("Solution using Cholesky method:")
    for i, sol in enumerate(solution_cholesky):
        print(f"x{i + 1} = {sol}")

except ValueError:
    # Solve using Doolittle method
    print("Matrix A is not symmetric positive definite. Using Doolittle method instead.")
    solution_doolittle = solve_linear_equations_doolittle(A, B)
    print("Solution using Doolittle method:")
    for i, sol in enumerate(solution_doolittle):
        print(f"x{i + 1} = {sol}")
