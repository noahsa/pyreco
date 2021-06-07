import collections

import pyreco.pyreco
import hts.functions
import numpy as np
import pandas as pd
import numpy.testing


def test_get_divisors():
    output = pyreco.pyreco.get_divisors(12)


    expected_output = [1, 2, 3, 4, 6, 12]

    assert (list(output) == expected_output)


def test_foreco_example(foreco_data):
    aggregation_matrix = np.array([[1,1,1,1,1],
                                   [1,1,0,0,0],
                                   [0,0,1,1,0]])
    summing_matrix = np.vstack((aggregation_matrix, np.identity(aggregation_matrix.shape[1])))

    basef = foreco_data.to_numpy()[0:, 1:]

    reconciled = pyreco.pyreco._octrec(basef=basef,
                                       m=12,
                                       summing_matrix=summing_matrix)

    foreco_data = foreco_data.set_index('Unnamed: 0')

    reconciled = np.reshape(reconciled, (-1, foreco_data.shape[1]))

    reconciled = pd.DataFrame(data=reconciled[0:, 0:],
                                 index=foreco_data.index,
                                 columns=foreco_data.columns)

    # Check temporal coherence
    yearly_forecast = (reconciled['k12_h1']).values
    semiannual_forecast = (reconciled['k6_h1'] + reconciled['k6_h2']).values
    triannual_forecast = (reconciled['k4_h1'] + reconciled['k4_h2'] + reconciled['k4_h3']).values
    quarterly_forecast = (reconciled['k3_h1'] + reconciled['k3_h2'] + reconciled['k3_h3'] + reconciled['k3_h4']).values
    bimonthly_forecast = (reconciled['k2_h1'] + reconciled['k2_h2'] + reconciled['k2_h3'] + \
                          reconciled['k2_h4'] + reconciled['k2_h5'] + reconciled['k2_h6']).values
    monthly_forecast = (reconciled['k1_h1'] + reconciled['k1_h2'] + reconciled['k1_h3'] + reconciled['k1_h4'] + \
                       reconciled['k1_h5'] + reconciled['k1_h6'] + reconciled['k1_h7'] + reconciled['k1_h8'] + \
                       reconciled['k1_h9'] + reconciled['k1_h10'] + reconciled['k1_h11'] + reconciled['k1_h12']).values
    numpy.testing.assert_almost_equal(yearly_forecast, semiannual_forecast)
    numpy.testing.assert_almost_equal(yearly_forecast, triannual_forecast)
    numpy.testing.assert_almost_equal(yearly_forecast, quarterly_forecast)
    numpy.testing.assert_almost_equal(yearly_forecast, bimonthly_forecast)
    numpy.testing.assert_almost_equal(yearly_forecast, monthly_forecast)


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

    reconciled = pyreco.pyreco.octrec(forecasts=forecasts,
                                      m=12,
                                      summing_matrix=sum_mat)
    # Check temporal coherence
    yearly_forecast = (reconciled['2020-Y1']).values
    semiannual_forecast = (reconciled['2020-S1'] + reconciled['2020-S2']).values
    triannual_forecast = (reconciled['2020-T1'] + reconciled['2020-T2'] + reconciled['2020-T3']).values
    quarterly_forecast = (reconciled['2020-Q1'] + reconciled['2020-Q2'] + reconciled['2020-Q3'] + reconciled['2020-Q4']).values
    bimonthly_forecast = (reconciled['2020-B1'] + reconciled['2020-B2'] + reconciled['2020-B3'] + \
                          reconciled['2020-B4'] + reconciled['2020-B5'] + reconciled['2020-B6']).values
    monthly_forecast = (reconciled['2020-01'] + reconciled['2020-02'] + reconciled['2020-03'] + reconciled['2020-04'] + \
                       reconciled['2020-05'] + reconciled['2020-06'] + reconciled['2020-07'] + reconciled['2020-08'] + \
                       reconciled['2020-09'] + reconciled['2020-10'] + reconciled['2020-11'] + reconciled['2020-12']).values
    numpy.testing.assert_almost_equal(yearly_forecast, semiannual_forecast)
    numpy.testing.assert_almost_equal(yearly_forecast, triannual_forecast)
    numpy.testing.assert_almost_equal(yearly_forecast, quarterly_forecast)
    numpy.testing.assert_almost_equal(yearly_forecast, bimonthly_forecast)
    numpy.testing.assert_almost_equal(yearly_forecast, monthly_forecast)



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
    reconciled = pyreco.pyreco.octrec(forecasts=forecasts,
                                      m=12,
                                      summing_matrix=sum_mat,
                                      kset=kset)

    # Check temporal coherence
    yearly_forecast = (reconciled['2020-Y1']).values
    quarterly_forecast = (reconciled['2020-Q1'] + reconciled['2020-Q2'] + reconciled['2020-Q3'] + reconciled['2020-Q4']).values
    monthly_forecast = (reconciled['2020-01'] + reconciled['2020-02'] + reconciled['2020-03'] + reconciled['2020-04'] + \
                       reconciled['2020-05'] + reconciled['2020-06'] + reconciled['2020-07'] + reconciled['2020-08'] + \
                       reconciled['2020-09'] + reconciled['2020-10'] + reconciled['2020-11'] + reconciled['2020-12']).values
    numpy.testing.assert_almost_equal(yearly_forecast, quarterly_forecast)
    numpy.testing.assert_almost_equal(yearly_forecast, monthly_forecast)
