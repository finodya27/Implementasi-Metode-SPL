import numpy as np

def main():
    while True:
        print("\nImplementasi Sistem Persamaan Linear")
        print("1. Metode Matriks Balikan")
        print("2. Metode Dekomposisi LU Gauss")
        print("3. Metode Dekomposisi Crout")

        method = int(input("Pilih Metode (1-3): "))
        if method == 1:
            solve_with_method("Matriks Balikan", np.linalg.inv)
        elif method == 2:
            solve_with_method("Dekomposisi LU Gauss", gauss_decomposition)
        elif method == 3:
            solve_with_method("Dekomposisi Crout", crout_decomposition)
        else:
            print("Error")

def solve_with_method(method_name, solver_func):
    n = int(input("Ordo Matriks: "))
    X = np.zeros((n, n))

    print("Masukkan matriks X:")
    for i in range(n):
        for j in range(n):
            X[i, j] = float(input(f"Matriks X[{i+1},{j+1}]: "))

    Y = np.zeros(n)
    print("\nMasukkan matriks Y:")
    for i in range(n):
        Y[i] = float(input(f"Matriks Y[{i+1}]: "))

    if method_name == "Matriks Balikan":
        X_inv = solver_func(X)
        x = np.dot(X_inv, Y)
    else:
        L, U = solver_func(X)
        y = forward_substitution(L, Y)
        x = backward_substitution(U, y)

    print("Hasil Akhir:")
    for i in range(n):
        print(f"x[{i+1}] = {x[i]}")

def gauss_decomposition(X):
    n = len(X)
    L = np.eye(n)
    U = X.copy()

    for k in range(n-1):
        if U[k, k] == 0:
            raise ValueError("Error")
        for i in range(k+1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    if U[n-1, n-1] == 0:
        raise ValueError("Error")
    return L, U

def crout_decomposition(X):
    n = len(X)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
        U[j, j] = 1  

        for i in range(j, n):
            sum_l = sum(L[i, k] * U[k, j] for k in range(i))
            L[i, j] = X[i, j] - sum_l

        for i in range(j, n):
            sum_u = sum(L[j, k] * U[k, i] for k in range(j))
            if L[j, j] == 0:
                raise ValueError("Error")
            U[j, i] = (X[j, i] - sum_u) / L[j, j]

    return L, U

def forward_substitution(L, Y):
    n = len(Y)
    y = np.zeros(n)

    for i in range(n):
        y[i] = (Y[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

if __name__ == "__main__":
    main()
