import numpy as np
import time

# Вариант 3
A = np.array([
    [11.72, -2.15,  1.26, 0.91],
    [ 1.33, 12.84,  1.48, 2.17],
    [-1.74,  1.21, 10.96, 1.58],
    [ 2.03, -1.16,  0.87,13.41]
], dtype=float)

b = np.array([12.84, 19.63, 13.47, 21.35], dtype=float)

def gauss(A, b):
    """Метод Гаусса с частичным выбором главного элемента."""
    M = np.hstack((A.copy(), b.reshape(-1, 1)))
    n = len(b)

    # Прямой ход
    for k in range(n - 1):
        p = k + np.argmax(np.abs(M[k:, k]))

        if abs(M[p, k]) < 1e-15:
            raise ValueError("Система вырожденная или почти вырожденная")

        if p != k:
            M[[k, p]] = M[[p, k]]

        for i in range(k + 1, n):
            m = M[i, k] / M[k, k]
            M[i, k:] -= m * M[k, k:]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (
            M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:n])
        ) / M[i, i]

    return x

def simple_iteration(A, b, x0, eps=1e-6, kmax=10000, verbose=True):
    """Метод простой итерации (Якоби)."""
    x = x0.copy().astype(float)

    for k in range(1, kmax + 1):
        xn = np.zeros_like(x)

        for i in range(len(b)):
            s = (
                np.dot(A[i, :i], x[:i])
                + np.dot(A[i, i + 1:], x[i + 1:])
            )
            xn[i] = (b[i] - s) / A[i, i]

        err = np.linalg.norm(xn - x)

        if verbose:
            print(f"k={k:2d}: x={xn}, eps_k={err:.10e}")

        x = xn

        if err < eps:
            return x, k

    print("Критерий остановки не выполнен.")
    return x, kmax

def seidel(A, b, x0, eps=1e-6, kmax=10000, verbose=True):
    """Метод Зейделя."""
    x = x0.copy().astype(float)

    for k in range(1, kmax + 1):
        old = x.copy()

        for i in range(len(b)):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], old[i + 1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]

        err = np.linalg.norm(x - old)

        if verbose:
            print(f"k={k:2d}: x={x}, eps_k={err:.10e}")

        if err < eps:
            return x, k

    print("Критерий остановки не выполнен.")
    return x, kmax

def residual(A, x, b):
    """Норма невязки ||Ax-b||."""
    return np.linalg.norm(A @ x - b)

def rel_error(x, x_lib):
    """Относительная ошибка ||x-x_lib|| / ||x_lib||."""
    return np.linalg.norm(x - x_lib) / np.linalg.norm(x_lib)

def print_result(name, x, x_lib):
    print(f"\n{name}")
    print("x =", x)
    print("||Ax-b|| =", residual(A, x, b))
    print("||x-x_lib|| =", np.linalg.norm(x - x_lib))
    print("delta =", rel_error(x, x_lib))

# 1. Контрольное решение NumPy
x_lib = np.linalg.solve(A, b)
print("||A x_lib - b|| =", residual(A, x_lib, b))

print("ИСХОДНАЯ СИСТЕМА")
print("A =\n", A)
print("b =", b)
print("\ndet(A) =", np.linalg.det(A))
print("cond(A) =", np.linalg.cond(A))
print("Контрольное решение NumPy:")
print(x_lib)

# 2. Проверка диагонального преобладания
print("\nДИАГОНАЛЬНОЕ ПРЕОБЛАДАНИЕ")
for i in range(len(b)):
    diagonal = abs(A[i, i])
    off_diagonal = sum(abs(A[i, j]) for j in range(len(b)) if j != i)
    print(f"Строка {i+1}: |a_ii|={diagonal:.2f}, сумма остальных={off_diagonal:.2f}")

# 3. Спектральный радиус матрицы итераций
D = np.diag(np.diag(A))
B = -np.linalg.inv(D) @ (A - D)
rho = max(abs(np.linalg.eigvals(B)))
print("\nМатрица B для простой итерации:\n", B)
print("rho(B) =", rho)

# 4. Метод Гаусса
x_g = gauss(A, b)
print_result("Метод Гаусса", x_g, x_lib)

# 5. Начальные приближения
initials = {
    "нулевое": np.zeros(4),
    "единичное": np.ones(4),
    "произвольное": np.array([1., 2., 1., 2.])
}

print("\nЭКСПЕРИМЕНТ С НАЧАЛЬНЫМИ ПРИБЛИЖЕНИЯМИ")
for name, x0 in initials.items():
    print(f"\n--- {name} начальное приближение ---")

    print("\nМетод простой итерации:")
    x, k = simple_iteration(A, b, x0, 1e-6, 10000, True)
    print_result(f"Простая итерация, {name}", x, x_lib)
    print("Число итераций:", k)

    print("\nМетод Зейделя:")
    x, k = seidel(A, b, x0, 1e-6, 10000, True)
    print_result(f"Зейдель, {name}", x, x_lib)
    print("Число итераций:", k)

# 6. Эксперимент с точностью
print("\nЭКСПЕРИМЕНТ С РАЗЛИЧНОЙ ТОЧНОСТЬЮ")
for eps in [1e-2, 1e-4, 1e-6]:
    print(f"\neps = {eps:.0e}")

    t0 = time.perf_counter()
    x, k = simple_iteration(A, b, np.zeros(4), eps, 10000, False)
    t = time.perf_counter() - t0
    print("Простая итерация:")
    print("x =", x)
    print("N =", k)
    print("||Ax-b|| =", residual(A, x, b))
    print(f"время = {t:.3e} с")

    t0 = time.perf_counter()
    x, k = seidel(A, b, np.zeros(4), eps, 10000, False)
    t = time.perf_counter() - t0
    print("Зейдель:")
    print("x =", x)
    print("N =", k)
    print("||Ax-b|| =", residual(A, x, b))
    print(f"время = {t:.3e} с")

# 7. Исследование округления коэффициентов
print("\nОКРУГЛЕНИЕ КОЭФФИЦИЕНТОВ")
for digits in [0, 1, 2, 4]:
    A_round = np.round(A, digits)
    b_round = np.round(b, digits)
    x_round = np.linalg.solve(A_round, b_round)

    print(f"\nОкругление до {digits} знаков:")
    print("x_round =", x_round)
    print("||x_round - x_lib|| =", np.linalg.norm(x_round - x_lib))
    print("det(A_round) =", np.linalg.det(A_round))
    print("cond(A_round) =", np.linalg.cond(A_round))