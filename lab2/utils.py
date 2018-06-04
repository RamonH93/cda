import pandas as pd
import argparse
def import_datasets():
    # set pandas output width
    desired_width = 320
    pd.set_option('display.width', desired_width)

    # read datasets
    df_trn1 = pd.read_csv('BATADAL_dataset03.csv', nrows=8762)
    df_trn2 = pd.read_csv('BATADAL_dataset04.csv')
    #df_test = pd.read_csv('BATADAL_test_dataset.csv')

    df_test = pd.read_csv('Batadal_labeled_test.csv', delimiter=";")

    # clean datasets
    df_trn1['DATETIME'] = pd.to_datetime(df_trn1['DATETIME'], dayfirst=True)  # convert timestamps
    df_trn2['DATETIME'] = pd.to_datetime(df_trn2['DATETIME'], dayfirst=True)  # convert timestamps
    df_trn2.rename(columns=lambda x: x.lstrip(), inplace=True) # remove whitespaces in headers
    df_test['DATETIME'] = pd.to_datetime(df_test['DATETIME'], dayfirst=True)  # convert timestamps

    return df_trn1, df_trn2, df_test
