import collections

import ctfr.ctfr
import hts.functions
import numpy as np
import pandas as pd


def test_get_divisors():
    output = ctfr.ctfr.get_divisors(12)


    expected_output = [1, 2, 3, 4, 6, 12]

    assert (list(output) == expected_output)


def test_foreco_example(foreco_data):
    aggregation_matrix = np.array([[1,1,1,1,1],
                                   [1,1,0,0,0],
                                   [0,0,1,1,0]])
    summing_matrix = np.vstack((aggregation_matrix, np.identity(aggregation_matrix.shape[1])))

    basef = foreco_data.to_numpy()[0:, 1:]

    reconciled = ctfr.ctfr._octrec(basef=basef,
                                   m=12,
                                   summing_matrix=summing_matrix)

    foreco_data = foreco_data.set_index('Unnamed: 0')

    reconciled = np.reshape(reconciled, (-1, foreco_data.shape[1]))

    reconciled_df = pd.DataFrame(data=reconciled[0:, 0:],
                                 index=foreco_data.index,
                                 columns=foreco_data.columns)


def test_all_frequencies(yearly_forecasts, semiannual_forecasts,
                         triannual_forecasts, quarterly_forecasts,
                         bimonthly_forecasts, monthly_forecasts):
    hier_df = pd.DataFrame(
        data={
            'ds': ['2020-01', '2020-02'] * 5,
            'lev1': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'lev2': ['X', 'X', 'Y', 'Y', 'Z', 'Z', 'X', 'X', 'Y', 'Y'],
            'val': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    level_names = ['lev1', 'lev2']
    hierarchy = [['lev1'], ['lev2']]
    gts_df, sum_mat, sum_mat_labels = hts.functions.get_hierarchichal_df(
        hier_df,
        level_names=level_names,
        hierarchy=hierarchy,
        date_colname='ds',
        val_colname='val',
    )

    forecasts = collections.OrderedDict({'yearly': yearly_forecasts,
                                         'semiannual': semiannual_forecasts,
                                         'triannual': triannual_forecasts,
                                         'quarterly': quarterly_forecasts,
                                         'bimonthly': bimonthly_forecasts,
                                         'monthly_forecasts': monthly_forecasts
                                        })

    reconciled = ctfr.ctfr.octrec(forecasts=forecasts,
                                  m=12,
                                  summing_matrix=sum_mat)
    reconciled



def test_less_frequencies(yearly_forecasts, quarterly_forecasts, monthly_forecasts):
    hier_df = pd.DataFrame(
        data={
            'ds': ['2020-01', '2020-02'] * 5,
            'lev1': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'lev2': ['X', 'X', 'Y', 'Y', 'Z', 'Z', 'X', 'X', 'Y', 'Y'],
            'val': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    level_names = ['lev1', 'lev2']
    hierarchy = [['lev1'], ['lev2']]
    gts_df, sum_mat, sum_mat_labels = hts.functions.get_hierarchichal_df(
        hier_df,
        level_names=level_names,
        hierarchy=hierarchy,
        date_colname='ds',
        val_colname='val',
    )

    forecasts = collections.OrderedDict({'yearly': yearly_forecasts,
                                         'quarterly': quarterly_forecasts,
                                         'monthly_forecasts': monthly_forecasts
                                        })

    kset = [1, 4, 12]
    reconciled = ctfr.ctfr.octrec(forecasts=forecasts,
                                  m=12,
                                  summing_matrix=sum_mat,
                                  kset=kset)
    reconciled
