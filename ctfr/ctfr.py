import numpy as np

def ctfr(basef: np.ndarray,
         m: int):
    tools = thf_tools(m)
    n = basef.shape[0]  # Number of rows in base forecasts
    C = ... # Summing matrix w/o idenity matrix
    nb = C.shape[0]  # N Cols/ ie num bottom level
    na = C.shape[1]  # N rows/ ie num agg

    Ut = np.hstack((np.identity(na), -C))
    return

def thf_tools(m: int,
              h : int = 1):
    kset = list(get_divisors(m))
    kset.reverse()

    p = len(kset)
    ks = sum(kset) - m
    kt = sum(kset)

    rev_kset = kset[1:][::-1]
    p_kset = kset[:-1]
    K = np.kron(np.identity(rev_kset[0] * h), np.ones(p_kset[0]))
    for i in range(1, len(kset[:-1])):
        res = np.kron(np.identity(rev_kset[i] * h), np.ones(p_kset[i]))
        K = np.vstack((K, res))
   # K = ...
    Zt = np.hstack((np.identity(h * ks), -K)) # Maybe hstack
    R = np.vstack((K, np.identity(m * h)))

    return {'K': K,
            'Zt': Zt,
            'R': R,
            'kset': kset,
            'p': p,
            'ks': ks,
            'kt': kt}


def get_divisors(n):
    for i in range(1, int(n / 2) + 1):
        if n % i == 0:
            yield i
    yield n
