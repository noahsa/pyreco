import pyreco.datasets
import pandas as pd
import pytest


@pytest.fixture(scope='session')
def yearly_forecasts():
    df = pd.DataFrame(index=['2020-Y1'],
                      data=[[1]* 11],
                      columns=['total', "A", "B", "X", "Y", "Z", "A_X", "A_Y", "A_Z", "B_X", "B_Y" ])
    return df


@pytest.fixture(scope='session')
def semiannual_forecasts():
    df = pd.DataFrame(index=['2020-S1', '2020-S2'],
                      data=[[1]* 11],
                      columns=['total', "A", "B", "X", "Y", "Z", "A_X", "A_Y", "A_Z", "B_X", "B_Y" ])
    return df


@pytest.fixture(scope='session')
def triannual_forecasts():
    df = pd.DataFrame(index=['2020-T1', '2020-T2', '2020-T3'],
                      data=[[1]* 11],
                      columns=['total', "A", "B", "X", "Y", "Z", "A_X", "A_Y", "A_Z", "B_X", "B_Y" ])
    return df


@pytest.fixture(scope='session')
def quarterly_forecasts():
    df = pd.DataFrame(index=['2020-Q1', '2020-Q2', '2020-Q3', '2020-Q4'],
                      data=[[1]* 11],
                      columns=['total', "A", "B", "X", "Y", "Z", "A_X", "A_Y", "A_Z", "B_X", "B_Y" ])
    return df


@pytest.fixture(scope='session')
def bimonthly_forecasts():
    df = pd.DataFrame(index=['2020-B1', '2020-B2', '2020-B3', '2020-B4', '2020-B5', '2020-B6'],
                      data=[[1]* 11],
                      columns=['total', "A", "B", "X", "Y", "Z", "A_X", "A_Y", "A_Z", "B_X", "B_Y" ])
    return df


@pytest.fixture(scope='session')
def monthly_forecasts():
    df = pd.DataFrame(index=['2020-01', '2020-02', '2020-03', '2020-04',
                             '2020-05', '2020-06', '2020-07', '2020-08',
                             '2020-09', '2020-10', '2020-11', '2020-12'],
                      data=[[1]* 11],
                      columns=['total', "A", "B", "X", "Y", "Z", "A_X", "A_Y", "A_Z", "B_X", "B_Y" ])
    return df


@pytest.fixture(scope='session')
def foreco_data():
    return pyreco.datasets.load_foreco()
