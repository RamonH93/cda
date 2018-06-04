import pandas as pd


def import_datasets():
    # set pandas output width
    desired_width = 320
    pd.set_option('display.width', desired_width)

    # read datasets
    df_trn1 = pd.read_csv('datasets\BATADAL_dataset03.csv')
    df_trn2 = pd.read_csv('datasets\BATADAL_dataset04.csv')
    df_test = pd.read_csv('datasets\Batadal_labeled_test.csv', delimiter=';')

    # clean datasets
    df_trn1['DATETIME'] = pd.to_datetime(df_trn1['DATETIME'], dayfirst=True)  # convert timestamps
    df_trn2['DATETIME'] = pd.to_datetime(df_trn2['DATETIME'], dayfirst=True)  # convert timestamps
    df_trn2.columns = [x.lstrip() for x in df_trn2.columns]  # remove whitespaces in headers
    df_test['DATETIME'] = pd.to_datetime(df_test['DATETIME'], dayfirst=True)  # convert timestamps

    return df_trn1, df_trn2, df_test


def get_model_loc(col, p, q):
    return 'ARMA_models\\' + col + '_' + str(p) + '.' + str(q) + '_model.pkl'


def get_test_model_loc(col, p, q):
    return 'ARMA_models\\' + 'test_' + col + '_' + str(p) + '.' + str(q) + '_model.pkl'


def get_model_params(model_loc):
    col = model_loc[12:-14]
    p = model_loc[-13:-12]
    q = model_loc[-11:-10]
    return col, p, q


def get_test_model_params(model_loc):
    col = model_loc[17:-14]
    p = model_loc[-13:-12]
    q = model_loc[-11:-10]
    return col, p, q
