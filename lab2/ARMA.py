import utils
import os
import math
import pickle
import pandas as pd
import patsy  # required for statsmodels
from statsmodels.tsa.arima_model import ARMA, ARMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from timeit import default_timer
from statsmodels.tools.parallel import parallel_func


def plot_acf_pacf(df, plot=True):
    if plot:
        for col in df:
            if col == 'DATETIME':
                continue

            try:
                plot_acf(df[col], lags=15)
                plt.title('Autocorrelation ' + col)

                plot_pacf(df[col], lags=15)
                plt.title('Partial Autocorrelation ' + col)
            except Exception, e:
                pass
        plt.show()


def mp_fitting(df, col, p, q):
    model_loc = utils.get_model_loc(col, p, q)
    if 'ATT_FLAG' in list(df):
        model_loc = utils.get_test_model_loc(col, p, q)
    if not os.path.isfile(model_loc) and col != 'DATETIME':
        try:
            starttime = default_timer()
            model = ARMA(endog=df[col], order=(int(p), int(q)), dates=df['DATETIME'], freq='H').fit(disp=0)
            aic = model.aic

            if math.isnan(aic):
                raise ValueError('aic is NaN')

            model.save(model_loc)

            print col + '(p=' + str(p) + ', q=' + str(q) + ') AIC=' + '{0:10.2f}'.format(
                aic) + ' execution time: ' + '{0:.3f}'.format((default_timer() - starttime))
        except Exception, e:
            print col + '(p=' + str(p) + ', q=' + str(q) + ') params caused an error: ' + str(e)
    else:
        print col + '(p=' + str(p) + ', q=' + str(q) + ') is skipped'


def select_best_models(columns):
    best_models = dict()
    for col in columns:
        try:
            best_p, best_q = select_x_best_model(col, 1)
            best_models[col] = [best_p, best_q]
        except IndexError:
            pass
    return best_models


def select_x_best_model(col, x):
    scores = list()
    filelist = [f for f in os.listdir('ARMA_models') if not f.startswith("test")]
    for f in filelist:
        model_loc = os.path.join('ARMA_models', f)
        fcol, p, q = utils.get_model_params(model_loc)
        if fcol == col:
            model = ARMAResults.load(model_loc)
            scores.append([p, q, model.aic])
    if x > len(scores):
        raise IndexError('index out of bounds')
    scores = sorted(scores, key=lambda x: x[2])
    return scores[x-1][0], scores[x-1][1]


def get_predictions_residuals(df, model_loc):
    if (model_loc is not None) and os.path.isfile(model_loc):
        model = ARMAResults.load(model_loc)
        col, p, q = utils.get_model_params(model_loc)
        if 'test' in model_loc:
            col, p, q = utils.get_test_model_params(model_loc)

        predictions = model.predict(0, len(df[col])-1)
        residuals = [df[col][i] - predictions[i] for i in range(len(df[col]))]

        return predictions, residuals
    return None, None


if __name__ == '__main__':

    # monitor execution time
    start = default_timer()

    df_train, _, df_test = utils.import_datasets()

    # drop discrete binary columns
    df_train.drop(['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9',
                  'S_PU10', 'S_PU11', 'S_V2', 'ATT_FLAG'], inplace=True, axis=1)
    df_test.drop(['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9',
                  'S_PU10', 'S_PU11', 'S_V2'], inplace=True, axis=1) # don't remove attack labels here

    # determine which p and q ranges to fit, by looking when the correlation first crosses the confidence interval
    # set plot=True to display the graphs for each sensor
    plot_acf_pacf(df_train, plot=False)
    p_range = 6
    q_range = 13

    # if they don't already exist, fit and save all possible training models with multiprocessing
    # commented out to save runtime, because the models are already generated
    '''
    parallel, p_func, n_jobs = parallel_func(mp_fitting, n_jobs=4, verbose=0)
    parallel(p_func(df_train, col, p, q) for col in df_train for p in xrange(p_range) for q in xrange(q_range))
    '''

    # compare aic scores to select the models with the best p and q parameters
    best_ARMA_models = dict()
    if(os.path.isfile('best_ARMA_models.pickle')):
        with open('best_ARMA_models.pickle', 'rb') as handle:
            best_ARMA_models = pickle.loads(handle.read())
    else:
        with open('best_ARMA_models.pickle', 'wb') as handle:
            best_ARMA_models = select_best_models(list(df_train))
            pickle.dump(best_ARMA_models, handle)


    # fit best models on test data, commented out to save runtime
    '''
    parallel, p_func, n_jobs = parallel_func(mp_fitting, n_jobs=4, verbose=0)
    parallel(p_func(df_test, col, p, q) for col in best_ARMA_models for p in best_ARMA_models[col][0] for q in best_ARMA_models[col][1])
    '''

    # fit next best parameters if best model failed to fit on test data, commented out to save runtime
    '''
    for col in best_ARMA_models:
        model_exists = tried_all_models = False
        i = 1
        while not model_exists and not tried_all_models:
            filelist = [f for f in os.listdir('ARMA_models') if f.startswith("test")]
            for f in filelist:
                if col in f:
                    model_exists = True
                    print col + ': fitted ' + str(i) + 'th best model on test data'
                    break
            if not model_exists:
                i += 1
                try:
                    p, q = select_x_best_model(col, i)
                    mp_fitting(df_test, col, p, q)
                except IndexError:
                    tried_all_models = True
                    print col + ': failed to fit model on test data'
    '''


    # get min and max residual errors of training models with params that were fitted on the test data
    filelist = [f for f in os.listdir('ARMA_models') if f.startswith("test")]
    errs_min_max = dict()
    for f in filelist:
        test_model_loc = os.path.join('ARMA_models', f)
        col, p, q = utils.get_test_model_params(test_model_loc)
        train_model_loc = utils.get_model_loc(col, p, q)
        _, residuals = get_predictions_residuals(df_train, train_model_loc)
        errs_min_max[col] = [min(residuals), max(residuals)]

    # find performance on test data
    alerts = {'DATETIME': df_test['DATETIME'].values, 'PRED': [0 for i in xrange(len(df_test['DATETIME']))],
              'ATT_FLAG': df_test['ATT_FLAG'].values}
    alerts = pd.DataFrame(data=alerts)
    filelist = [f for f in os.listdir('ARMA_models') if f.startswith("test")]
    for f in filelist:
        model_loc = os.path.join('ARMA_models', f)
        col, p, q = utils.get_test_model_params(model_loc)

        predictions, residuals = get_predictions_residuals(df_test, model_loc)
        for n in xrange(len(residuals)):
            if residuals[n] < errs_min_max[col][0] or residuals[n] > errs_min_max[col][1]:
                alerts.loc[n, 'PRED'] = 1

        # display sensor data and ARMA prediction
        '''
        pred_label = col + ' ARMA prediction: p=' + p + ', q=' + q
        plt.plot(df_test['DATETIME'], df_test[col])
        plt.plot(df_test['DATETIME'], predictions, 'r--', label=pred_label)
        plt.title(col)
        plt.legend()
        plt.show()
        '''

        # display residual errors and thresholds
        '''
        # plt.plot(df_test['DATETIME'], residuals)
        # plt.plot(df_test['DATETIME'], [errs_min_max[col][0] for i in xrange(len(df_test['DATETIME']))], 'g--')
        # plt.plot(df_test['DATETIME'], [errs_min_max[col][1] for i in xrange(len(df_test['DATETIME']))], 'g--')
        # plt.title(col)
        # plt.show()
        '''

    tp = fp = fn = tn = 0
    for i in xrange(len(alerts['DATETIME'])):
        if alerts.loc[i, 'PRED'] == 1:
            if alerts.loc[i, 'ATT_FLAG'] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if alerts.loc[i, 'ATT_FLAG'] == 1:
                fn += 1
            else:
                tn += 1

    accuracy = float(tp + tn) / (tp + tn + fn + fp)
    recall = float(tp) / (tp + fn)
    precision = float(tp) / (tp + fp)

    print '{0:7}'.format('ARMA predictions on testdata: ') + 'tp=' + str(tp) + ', fp=' + str(fp) + ', fn=' \
              + str(fn) + ', tn=' + str(tn), ' accuracy:', accuracy, ' recall:', recall, 'precision', precision

    print 'program execution time: ' + '{0:.3f}'.format((default_timer() - start))
