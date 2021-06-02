import numpy as np
import pandas as pd
import hts


def ctfr(basef: np.ndarray,
         m: int):
    hts = hts_tools()

    tmp = thf_tools(m)

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

def recoM(basef, W, Ht, b_pos, S):
    lm_dx = np.matmul(Ht, basef)
    lm_sx = np.matmul(np.matmul(Ht, W), Ht.T)
    recf = np.matmul(np.identity(W.shape[1]), basef) - np.matmul(np.matmul(W, Ht.T), np.linalg.solve(np.matrix(lm_sx, dtype='float'), np.matrix(lm_dx, dtype='float')))
    return recf


def hts_tools():
    # Should compute this with scikit-hts
    C = np.array([[1,1,1,1,1],
                 [1,1,0,0,0],
                 [0,0,1,1,0]])

    #  C = ... # Summing matrix w/o idenity matrix, aggregation
    nb = C.shape[1]  # N Cols/ ie num bottom level
    na = C.shape[0]  # N rows/ ie num agg

    n = na + nb

    S = np.vstack((C, np.identity(nb)))  # S is summing matrix

    Ut = np.hstack((np.identity(na), -C))

    return {'C': C,
            'nb': nb,
            'na': na,
            'n': n,
            'S': S,
            'Ut': Ut
            }


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

def commat(r: int,
           n: int):
    I = np.arange(1, (r * n) + 1, 1)
    I = np.reshape(I, (-1, r))
    P = np.identity(r * n)
    return

def commutation_matrix_sp(r, n):
    """
    https://stackoverflow.com/questions/60678746/compute-commutation-matrix-in-numpy-scipy-efficiently
    """
    from scipy.sparse import csr_matrix
    m, n = r, n
    row  = np.arange(m*n)
    col  = row.reshape((m, n), order='F').ravel()
    data = np.ones(m*n, dtype=np.int8)
    K = csr_matrix((data, (row, col)), shape=(m*n, m*n))
    return K


if __name__ == '__main__':
    basef = pd.read_csv('basef.csv')
    reconciled = ctfr(basef=basef.to_numpy()[0:, 1:],
                      m=12)

    # Reshape reconciled forecasts to put back into original DataFrame shape
    basef = basef.set_index('Unnamed: 0')
    rec = np.reshape(reconciled, (-1, basef.shape[1]))

    reconciled_df = pd.DataFrame(data=reconciled[0:, 0:],
                                 index=basef.index,
                                 columns=basef.columns)
