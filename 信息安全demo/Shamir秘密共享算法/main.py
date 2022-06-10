import numpy as np
import random
#求最大公约数
def gcd(a,b):
    if b==0: return a
    else: return gcd(b, a%b)

def findModReverse(a, m):  # 扩展欧几里得算法求模逆

        if gcd(a, m) != 1:
            return None
        u1, u2, u3 = 1, 0, a
        v1, v2, v3 = 0, 1, m
        while v3 != 0:
            q = u3 // v3
            v1, v2, v3, u1, u2, u3 = (u1 - q * v1), (u2 - q * v2), (u3 - q * v3), v1, v2, v3
        return u1 % m


def divresult(m):
    Mj = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(0, len(m)):

        for j in range(0, len(m)):
            if (i == j):
                Mj[i] = Mj[i] * 1

            else:
                Mj[i] = Mj[i] * m[j]
    return Mj
#求解N和M
def fun1(d,t):
    N=1
    M=1
    for i in range(0,t):
        N=N*d[i]
    for i in range(len(d)-t+1,len(d)):
        M=M*d[i]
    return N,M
def findk(d,k):
    k1=[1,1,1,1,1,1,1]
    for i in range(0,len(d)):
        k1[i]=k%d[i]
    k1=k1[0:len(d)]
    return k1

def ChineseSurplus(k,d,t):  #中国剩余定义求解方程
    m = d[0:t]
    a = k[0:t]
    flag = 1
    # Step1:计算连乘
    m1 = 1
    for i in range(0, len(m)):
        m1 = m1 * m[i]
    # Step2:计算Mj

    Mj = divresult(m)
    Mj1 = [0, 0, 0, 0, 0, 0, 0]
    # Step3:计算模的逆

    for i in range(0, len(m)):
        Mj1[i] = findModReverse(Mj[i], m[i])
    # 最后的x
    x = 0

    for i in range(0, len(m)):
        x = x + Mj[i] * Mj1[i] * a[i]

    result = x % m1
    return result



#定义d数组
#问题是如何产生合适的d值
def judge1(m, num):
        flag1 = 1
        for i in range(0, num):
            for j in range(0, num):
                if (gcd(m[i], m[j]) != 1) & (i != j):
                    flag1 = 0
                    break
        return flag1

#产生d数组
def find_d1():
    d = [1, 1, 1, 1, 1] #初始化d数组
    temp = random.randint(pow(10, 167), pow(10, 168))
    d[0] = temp
    i = 1
    while (i < 5):
        temp = random.randint(pow(10, 167), pow(10, 168))
        d[i] = temp
        if (judge1(d, i + 1) == 1):
            i = i + 1
    return d


#500位的大数作为秘密
k=2074722246773485207821695222107608587480996474721117292752992589912196684750549658310084416732550001130212021515151510511200515102155022515152074722246773485207821695222107608587480996474721117292752992589912196684750549658310084416732550001130212021515151510511200515102155022515152074722246773485207821695222107608587480996474721117292752992589912196684750549658310084416732550001130212021515151510511200515102155022515152074722246773485207821695222107608587480996474721117292752992589912196684750
#step1:生成符合条件的d值
d=find_d1()
print("d数组为:")
print(d)
#step2:计算N和M的值

print("N和M的值分别为:")
N,M=fun1(d,3)
print(N)
print(M)
#求k
k1=findk(d,k)
#利用中国剩余定理求解
result=ChineseSurplus(k1,d,3)
print("最后恢复的明文为:")
print(result)
