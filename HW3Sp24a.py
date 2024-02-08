import math
import copy
#I used ChatGPT to help me write this code. I am going to have it explain the code to me step
#by step so that I can better understand the code and why the code is written this way.
#I had to really work on giving the correct prompts so that the code would run the way
#it needed to run per the homework.

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
    if not is_symmetric(matrix):
        return False

    n = len(matrix)
    for i in range(n):
        submatrix = [[matrix[row][col] for col in range(i + 1)] for row in range(i + 1)]
        det_submatrix = determinant(submatrix)
        if det_submatrix <= 0:
            return False
    return True

def determinant(matrix):
    """
    Calculate the determinant of a matrix.

    Args:
    matrix (list of lists): The matrix to calculate the determinant of.

    Returns:
    float: The determinant of the matrix.
    """
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    elif n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det = 0
        for j in range(n):
            minor = [row[:j] + row[j + 1:] for row in matrix[1:]]
            det += matrix[0][j] * (-1) ** j * determinant(minor)
        return det

def cholesky_decomposition(matrix):
    """
    Perform Cholesky decomposition on a matrix.

    Args:
    matrix (list of lists): The matrix to decompose.

    Returns:
    list of lists: The lower triangular matrix of the Cholesky decomposition.
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

def solve_cholesky(matrix_A, vector_b):
    """
    Solve the matrix equation Ax = b using Cholesky method.

    Args:
    matrix_A (list of lists): The matrix A in the equation Ax = b.
    vector_b (list of lists): The vector b in the equation Ax = b.

    Returns:
    list: The solution vector x.
    """
    if not is_symmetric(matrix_A) or not is_positive_definite(matrix_A):
        return None
    L = cholesky_decomposition(matrix_A)
    n = len(matrix_A)
    y = [0.0] * n
    for i in range(n):
        sum_val = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (vector_b[i][0] - sum_val) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = sum(L[j][i] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_val) / L[i][i]
    return x

def lu_decomposition(matrix):
    """
    Perform LU decomposition using Doolittle method on a matrix.

    Args:
    matrix (list of lists): The matrix to decompose.

    Returns:
    tuple: A tuple containing the lower triangular matrix L and the upper triangular matrix U.
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    U = copy.deepcopy(matrix)
    for i in range(n):
        L[i][i] = 1.0
        for j in range(i + 1, n):
            factor = U[j][i] / U[i][i]
            L[j][i] = factor
            for k in range(i, n):
                U[j][k] -= factor * U[i][k]
    return L, U

def solve_doolittle(matrix_A, vector_b):
    """
    Solve the matrix equation Ax = b using LU decomposition (Doolittle method).

    Args:
    matrix_A (list of lists): The matrix A in the equation Ax = b.
    vector_b (list of lists): The vector b in the equation Ax = b.

    Returns:
    list: The solution vector x.
    """
    n = len(matrix_A)
    L, U = lu_decomposition(matrix_A)
    y = [0.0] * n
    for i in range(n):
        sum_val = sum(L[i][j] * y[j] for j in range(i))
        y[i] = vector_b[i][0] - sum_val
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_val) / U[i][i]
    return x

# Problem 1 matrices
matrix_A = [[1, -1, 3, 2], [-1, 5, -5, -2], [3, -5, 19, 3], [2, -2, 3, 21]]
vector_b = [[15], [-35], [94], [1]]

# Check if matrix_A is symmetric and positive definite
if is_symmetric(matrix_A) and is_positive_definite(matrix_A):
    print("Matrix A is symmetric and positive definite. Using Cholesky method.")
    solution = solve_cholesky(matrix_A, vector_b)
else:
    print("Matrix A is not symmetric and positive definite. Using Doolittle method.")
    solution = solve_doolittle(matrix_A, vector_b)

# Print solution
print("Solution vector x:")
print(solution)

# Problem 2 matrices
matrix_B = [[4, 2, 4, 0], [2, 2, 3, 2], [4, 3, 6, 3], [0, 2, 3, 9]]
vector_b_2 = [[20], [36], [60], [122]]

# Check if matrix_B is symmetric and positive definite
if is_symmetric(matrix_B) and is_positive_definite(matrix_B):
    print("Matrix B is symmetric and positive definite. Using Cholesky method.")
    solution_2 = solve_cholesky(matrix_B, vector_b_2)
else:
    print("Matrix B is not symmetric and positive definite. Using Doolittle method.")
    solution_2 = solve_doolittle(matrix_B, vector_b_2)

# Print solution
print("Solution vector x for Problem 2:")
print(solution_2)
