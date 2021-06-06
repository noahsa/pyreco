import collections
from typing import OrderedDict, List

import numpy as np
import pandas as pd
import scipy.sparse


def _octrec(basef: np.ndarray,
            m: int,
            summing_matrix: np.ndarray,
            kset: List[int] = None
            ) -> np.ndarray:
    hts = hts_tools(summing_matrix=summing_matrix)

    tmp = thf_tools(m=m, kset=kset)

    h = 1

    a = np.kron(hts['Ut'], np.identity(h * tmp['kt']))
    b = np.kron(np.identity(hts['n']), tmp['Zt'])
    Hbrevet = np.vstack((a, b))

    #P = commat(h * tmp['kt'], hts['n'])
    P = commutation_matrix_sp(h * tmp['kt'], hts['n']).toarray()

    Us = np.matmul(np.hstack((np.zeros((h * hts['Ut'].shape[0] * m, hts['n'] * h * tmp['ks'])), np.kron(np.identity(h * m), hts['Ut']))), P)

    Ht = np.vstack((Us, np.kron(np.identity(hts['n']), tmp['Zt'])))

    Ccheck = np.vstack((np.kron(hts['C'], tmp['R']),np.kron(np.identity(hts['nb']), tmp['K'])))
    Hcheckt = np.hstack((np.identity(h * (hts['na'] * m + hts['n'] * tmp['ks'])), -Ccheck))
    Scheck = np.vstack((Ccheck, np.identity(hts['nb'] * m * h)))
    Stilde = np.kron(hts['S'], tmp['R'])

    S = Stilde

    h = basef.shape[1] / tmp['kt']

    # Winging it that this is an identity matrix
    Dh = np.identity(np.sum(tmp['kset']) * hts['n'])

    Ybase = np.reshape(np.matmul(Dh, basef.flatten()), (-1, int(h)))

    Omega = np.identity(hts['n'] * tmp['kt'])

    b_pos = ([0] * (hts['na'] * tmp['kt'])) + [int(truth) for truth in list(np.array(list(np.repeat(tmp['kset'], tmp['kset'][::-1])) * hts['nb']) == 1)]

    reconciled = recoM(Ybase, Omega, Ht,b_pos, S)

    return reconciled


def octrec(forecasts: collections.OrderedDict[str, pd.DataFrame],
           summing_matrix: np.ndarray,
           m: int,
           kset: List[int] = None
           ):
    basef, column_names, idx = to_matrix_format(forecasts)

    reconciled = _octrec(basef=basef,
                         m=m,
                         summing_matrix=summing_matrix,
                         kset=kset)

    reconciled = np.reshape(reconciled, (-1, basef.shape[1]))

    reconciled_df = pd.DataFrame(data=reconciled[0:, 0:],
                                 index=idx,
                                 columns=column_names)
    return reconciled_df


def recoM(basef: np.ndarray,
          W,
          Ht,
          b_pos: List[int],
          S
          ):
    lm_dx = np.matmul(Ht, basef)
    lm_sx = np.matmul(np.matmul(Ht, W), Ht.T)
    recf = np.matmul(np.identity(W.shape[1]), basef) - np.matmul(np.matmul(W, Ht.T), np.linalg.solve(np.matrix(lm_sx, dtype='float'), np.matrix(lm_dx, dtype='float')))
    return recf


def hts_tools(summing_matrix: np.ndarray):
    n_columns = summing_matrix.shape[1]

    # Aggregation matrix
    C = summing_matrix[:-n_columns]

    nb = C.shape[1]
    na = C.shape[0]

    n = na + nb

    # Summing matrix
    S = np.vstack((C, np.identity(nb)))  # Could set to summing_matrix parameter

    Ut = np.hstack((np.identity(na), -C))

    return {'C': C,
            'nb': nb,
            'na': na,
            'n': n,
            'S': S,
            'Ut': Ut
            }


def thf_tools(m: int,
              h : int = 1,
              kset: List[int] = None
              ):
    if not kset:
        kset = list(get_divisors(m))
    kset.reverse()

    p = len(kset)

    ks = sum(kset) - m
    kt = sum(kset)


    rev_kset = kset[1:][::-1]
    p_kset = kset[:-1]
    K = np.kron(np.identity(rev_kset[0] * h), np.ones(p_kset[0]))
    for i in range(1, len(kset[:-1])):
        if kset:
            p_kset[i] = int(m / rev_kset[i])
        res = np.kron(np.identity(rev_kset[i] * h), np.ones(p_kset[i]))
        K = np.vstack((K, res))

    Zt = np.hstack((np.identity(h * ks), -K)) # Maybe hstack
    R = np.vstack((K, np.identity(m * h)))

    return {'K': K,
            'Zt': Zt,
            'R': R,
            'kset': kset,
            'p': p,
            'ks': ks,
            'kt': kt}


def get_divisors(n: int):
    for i in range(1, int(n / 2) + 1):
        if n % i == 0:
            yield i
    yield n


def commat(r: int,
           n: int
           ):
    I = np.arange(1, (r * n) + 1, 1)
    I = np.reshape(I, (-1, r))
    P = np.identity(r * n)
    return


def commutation_matrix_sp(r: int,
                          n: int
                          ):
    """
    https://stackoverflow.com/questions/60678746/compute-commutation-matrix-in-numpy-scipy-efficiently
    """
    m, n = r, n
    row  = np.arange(m*n)
    col  = row.reshape((m, n), order='F').ravel()
    data = np.ones(m*n, dtype=np.int8)
    K = scipy.sparse.csr_matrix((data, (row, col)), shape=(m*n, m*n))
    return K


def to_matrix_format(forecasts: collections.OrderedDict[str, pd.DataFrame]):
    basef = pd.concat([v.T for k,v in forecasts.items()],
                      axis=1,
                      join='inner')
    column_names = basef.columns
    idx = basef.index
    basef = basef.to_numpy()[0:, 0:]
    return basef, column_names, idx
