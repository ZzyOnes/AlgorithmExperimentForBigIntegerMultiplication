import time
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal


# 产生随机大整数
def getNum(N):
    a = 0
    b = 0
    while a == 0:
        a = np.random.randint(10)
    a = str(a)
    for _ in range(N - 1):
        a += str(np.random.randint(10))
    while b == 0:
        b = np.random.randint(10)
    b = str(b)
    for _ in range(N - 1):
        b += str(np.random.randint(10))
    return a, b


# 判断大整数大小,按 大、小返回
def compareTwoNum(x, y):
    n_x = len(x)
    n_y = len(y)
    if n_x > n_y:
        return x, y, 1
    if n_x < n_y:
        return y, x, -1
    for i in range(n_x):
        if x[i] == y[i]:
            continue
        elif x[i] > y[i]:
            return x, y, 1
        else:
            return y, x, -1
    return x, y, 1


# 大整数加法
def BigNumAdd(x, y):
    # res = ""
    # x = x[::-1]
    # y = y[::-1]
    # n_x = len(x)
    # n_y = len(y)
    # n = max(n_y, n_x)
    # pr = 0  # 进位
    # for i in range(n):
    #     if i > n_x - 1:
    #         a = 0
    #     else:
    #         a = int(x[i])
    #     if i > n_y - 1:
    #         b = 0
    #     else:
    #         b = int(y[i])
    #     num = a + b + pr
    #     if num >= 10:
    #         cur = num - 10
    #         pr = 1
    #     else:
    #         cur = num
    #         pr = 0
    #     res += str(cur)
    # if pr != 0:
    #     res += str(pr)
    # return res[::-1]
    if x == "":
        a = 0
    else:
        a = int(x)
    if y == "":
        b = 0
    else:
        b = int(y)
    return str(a + b)


# 大整数减法
def BigNumSub(x, y):
    # x, y, tag = compareTwoNum(x, y)
    # n_x = len(x)
    # n_y = len(y)
    # x = x[::-1]
    # y = y[::-1]
    # br = 0
    # res = ""
    # for i in range(n_x):
    #     a = int(x[i])
    #     if i > n_y - 1:
    #         b = 0
    #     else:
    #         b = int(y[i])
    #     a -= br
    #     if a >= b:
    #         br = 0
    #     else:
    #         br = 1
    #     cur = 10 * br + a - b
    #     res += str(cur)
    # # 去掉末尾多余的0
    # while res[-1] == "0":
    #     if len(res) == 1:
    #         break
    #     res = res[:-1]
    # if tag == -1:
    #     res += "-"
    # return res[::-1]
    if x == "":
        a = 0
    else:
        a = int(x)
    if y == "":
        b = 0
    else:
        b = int(y)
    return str(a - b)


def baseMul(x, power):
    power = int(power)
    n = len(x)
    res = ""
    pr = 0
    for i in range(n - 1, -1, -1):
        num = pr + power * int(x[i])
        cur = num % 10
        res += str(cur)
        pr = num // 10
    if pr != 0:
        res += str(pr)
    return res[::-1]


# 暴力大数乘法
def BigNumMul(x, y):
    n_y = len(y)
    res = ""
    for i in range(n_y - 1, -1, -1):
        res1 = baseMul(x, y[i]) + "0" * (n_y - 1 - i)
        if i == n_y - 1:
            res = res1
        else:
            res = BigNumAdd(res, res1)
    return res


# 普通递归大数乘法
def MutiProcess(x, y):
    n_x = len(x)
    n_y = len(y)
    if n_x < 2 and n_y < 2:
        return str(int(x) * int(y))
    if n_x < 2:
        return baseMul(y, x)
    if n_y < 2:
        return baseMul(x, y)
    mid_x = n_x // 2
    mid_y = n_y // 2
    a = x[:n_x - mid_x]
    b = x[n_x - mid_x:]
    c = y[:n_y - mid_y]
    d = y[n_y - mid_y:]
    res1 = MutiProcess(a, c) + "0" * (mid_x + mid_y)
    res2 = MutiProcess(a, d) + "0" * mid_x
    res3 = MutiProcess(b, c) + "0" * mid_y
    res4 = MutiProcess(b, d)
    res5 = BigNumAdd(res1, res2)
    res6 = BigNumAdd(res3, res4)
    res = BigNumAdd(res5, res6)
    return res


# 进阶版递归大数乘法
def karatsuba(x, y):
    n_x = len(x)
    n_y = len(y)
    if x == "" or y == "":
        return "0"
    if n_x < 2 and n_y < 2:
        return str(int(x) * int(y))
    if n_x < 2:
        return baseMul(y, x)
    if n_y < 2:
        return baseMul(x, y)
    n = n_x // 2
    a = x[:n_x - n]
    b = x[n_x - n:]
    c = y[:n_y - n]
    d = y[n_y - n:]
    res1 = karatsuba(a, c)
    res2 = karatsuba(b, d)
    res3 = BigNumAdd(a, b)
    res4 = BigNumAdd(c, d)
    res5 = karatsuba(res3, res4)
    res6 = BigNumSub(res5, res1)
    if res6[0] == '-':
        res7 = BigNumAdd(res6, res2)
        res7 = res7[::-1] + '-'
        res7 = res7[::-1]
    else:
        res7 = BigNumSub(res6, res2)
    res1 = res1 + "0" * 2 * n
    res7 = res7 + "0" * n
    res = BigNumAdd(res1, res7)
    res = BigNumAdd(res, res2)
    return res


def calc_all_number(D):
    result = 0
    for i in range(D.shape[0]):
        result += round(Decimal(D[i].real)) * 10 ** i
    return str(result)


# 计算离散傅里叶正变换：xk = ∑ xn * e**-(j*2pi*k*j/N)
def DFT_slow(xn):
    N = xn.shape[0]
    j = np.arange(N)
    k = j.reshape((N, 1))
    factor = np.exp(-2j * np.pi * k * j / N)
    return np.dot(factor, xn)


# 计算离散傅里叶逆变换：xn = 1 / N * ∑ xk * e**(j*2pi*k*j/N)
def IDFT_slow(xk):
    N = xk.shape[0]
    j = np.arange(N)
    k = j.reshape((N, 1))
    factor = np.exp(2j * np.pi * k * j / N)
    return 1 / N * np.dot(factor, xk)


def FFT(xn):
    N = xn.shape[0]
    N_min = min(N, 128)
    AX = DFT_slow(xn=xn.reshape((N_min, -1)))
    while AX.shape[0] < N:
        AX_even = AX[:, :int(AX.shape[1] / 2)]  # 偶数项多项式
        AX_odd = AX[:, int(AX.shape[1] / 2):]  # 奇数项多项式
        factor = np.exp(-2j * np.pi * np.arange(AX.shape[0]) / AX.shape[0] / 2)
        factor = factor.reshape((factor.shape[0], 1))
        factor_AX_odd = factor * AX_odd
        AX = np.vstack([AX_even + factor_AX_odd, AX_even - factor_AX_odd])
    return AX.ravel()


def IFFT(xk):
    N = xk.shape[0]
    N_min = min(N, 128)
    AX = N_min * IDFT_slow(xk=xk.reshape((N_min, -1)))
    while AX.shape[0] < N:
        AX_even = AX[:, :int(AX.shape[1] / 2)]
        AX_odd = AX[:, int(AX.shape[1] / 2):]
        factor = np.exp(2j * np.pi * np.arange(AX.shape[0]) / AX.shape[0] / 2)
        factor = factor.reshape((factor.shape[0], 1))
        factor_AX_odd = factor * AX_odd
        AX = np.vstack([AX_even + factor_AX_odd, AX_even - factor_AX_odd])
    return 1 / N * AX.ravel()


# 使用矩阵乘法版本：离散傅里叶变换
def DFT_multiply(N, x1_vec, x2_vec):
    A = DFT_slow(xn=np.array(x1_vec + [0] * (N - len(x1_vec))))
    B = DFT_slow(xn=np.array(x2_vec + [0] * (N - len(x2_vec))))
    C = A * B
    D = IDFT_slow(xk=C)
    return calc_all_number(D)


# 使用矩阵乘法版本：快速傅里叶变换
def FFT_multiply(N, x1_vec, x2_vec):
    A = FFT(xn=np.array(x1_vec + [0] * (N - len(x1_vec))))
    B = FFT(xn=np.array(x2_vec + [0] * (N - len(x2_vec))))
    C = A * B
    D = IFFT(xk=C)
    return calc_all_number(D)


# 使用自带的傅里叶变化
def numpy_FFT_multiply(N, x1_vec, x2_vec):
    A = np.fft.fft(np.array(x1_vec + [0] * (N - len(x1_vec))))
    B = np.fft.fft(np.array(x2_vec + [0] * (N - len(x2_vec))))
    C = A * B
    D = np.fft.ifft(C)
    return calc_all_number(D)


def TestAll():
    Num = [10, 100, 500, 1000, 5000, 10000, 100000]
    Time_BigNumMul = []
    Time_MutiProcess = []
    Time_karatsuba = []
    Time_py = []
    Time_FFT = []
    for n in Num:
        print("-------------开始计算规模为 %d的数------------" % n)
        a, b = getNum(n)
        x1_vec = list(map(int, list(reversed(a))))
        x2_vec = list(map(int, list(reversed(b))))
        # 1、加倍次数界：由分治法的思想，将两个多项式的次数界补全为2的幂次
        extra_len = 1
        while 2 ** extra_len < len(x1_vec) + len(x2_vec):
            extra_len += 1
        N = 2 ** extra_len
        # 2、求值：通过FFT计算出两个多项式的点值表达
        # 3、逐点相乘：将两个多项式的点值依次相乘，得到相乘结果的点值
        # 4、插值：通过IDFT计算相乘结果的点值，得到相乘结果的每一个系数
        startTime = time.time()
        res5 = numpy_FFT_multiply(N=N, x1_vec=x1_vec, x2_vec=x2_vec)
        endTime = time.time()
        runTime = (endTime - startTime)
        Time_FFT.append(runTime)
        print("FFT 已算完  耗时 %3f秒" % runTime)
        startTime = time.time()
        res1 = BigNumMul(a, b)
        endTime = time.time()
        runTime = (endTime - startTime)
        Time_BigNumMul.append(runTime)
        print("普通大整数乘法 已算完 耗时 %3f秒" % runTime)
        startTime = time.time()
        res2 = MutiProcess(a, b)
        endTime = time.time()
        runTime = (endTime - startTime)
        Time_MutiProcess.append(runTime)
        print("递归大整数乘法 已算完 耗时 %3f秒" % runTime)
        startTime = time.time()
        res3 = karatsuba(a, b)
        endTime = time.time()
        runTime = (endTime - startTime)
        Time_karatsuba.append(runTime)
        print("改进递归大整数乘法 已算完 耗时 %3f秒" % runTime)
        startTime = time.time()
        res4 = int(a) * int(b)
        endTime = time.time()
        runTime = (endTime - startTime)
        Time_py.append(runTime)
        if res1 != res2 or res3 != res2 or res3 != res1 or res1 != str(res4) or res5 != res1:
            print("运算结果错误！")
            return None
    return Num, Time_BigNumMul, Time_MutiProcess, Time_karatsuba, Time_py, Time_FFT


if __name__ == "__main__":
    N, Time_B, Time_M, Time_k, Time_p, Time_F = TestAll()  # 获得测试 规模、算术法、递归法、karatsuba法
    print("普通大整数乘法耗时 /s：", Time_B)
    print("递归乘法耗时 /s：", Time_M)
    print("改进递归乘法耗时 /s：", Time_k)
    print("FFT乘法耗时 /s：", Time_F)
    print("python自带乘法耗时 /s：", Time_p)
    plt.plot(np.log10(N), np.log10(Time_B), 'o--', color="blue")
    plt.plot(np.log10(N), np.log10(Time_M), 'o--', color="orange")
    plt.plot(np.log10(N), np.log10(Time_k), 'o--', color="green")
    plt.plot(np.log10(N), np.log10(Time_F), 'o--', color="red")
    plt.plot(np.log10(N), np.log10(Time_p), 'o--', color="purple")
    plt.legend(['arithmetic operations', 'recursion', 'karatsuba', "FFT", "python's own method"], loc='upper left')
    plt.xlabel("lg(number scale) /bit")
    plt.ylabel("lg(runtim) /s")
    plt.grid(b=True)
    plt.savefig("test8")
    plt.show()

