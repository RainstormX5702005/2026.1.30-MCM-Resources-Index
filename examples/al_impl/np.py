import timeit
import time
import threading
import numpy as np


def fibbonacci_by_recursion(n):
    if n <= 1:
        return n
    return fibbonacci_by_recursion(n - 1) + fibbonacci_by_recursion(n - 2)


# 使用 DP 优化的方法，递推公式为：a[i] = a[i-1] + a[i-2]
def fibbonacci_by_dp(n):
    if n <= 1:
        return n
    arr = np.zeros(n + 1, dtype=int)
    arr[0], arr[1] = 0, 1
    for i in range(2, n + 1):
        arr[i] = arr[i-1] + arr[i-2]
    return arr[n]


def comp_performance(n):
    rec_time = timeit.timeit(lambda: fibbonacci_by_recursion(n), number=10)
    dp_time = timeit.timeit(lambda: fibbonacci_by_dp(n), number=10)
    print(f"Recursion time for n={n}: {rec_time:.6f} seconds")
    print(f"Dynamic Programming time for n={n}: {dp_time:.6f} seconds")