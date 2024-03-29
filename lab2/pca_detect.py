import utils
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import discrete

from saxpy.alphabet import cuts_for_asize
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string

if __name__ == "__main__":
    df_trn1, df_trn2, df_test = utils.import_datasets()
    df_trn1 = df_trn1.drop(df_trn1.index[[1764,2461,3000,3001,3336,3337,5572,6084,7573,8646]])
    #df_trn1 = df_trn1.drop(df_trn1.index[[205,581,582,697,1052,1058,1185,1284,1565,1592,1744,1764,1891,2430,2461,2871,3000,3001,3200, 3301,3336,3337,3614,3628,3857,3880,3974,3988,4251,4623,5049,5149,5186,5271,5571,5572,5573,5724,6084,6614,6663,6765,6941,7573,8238,8295,8646]])
    df_time1 = df_trn1['DATETIME']
    df_time2 = df_test['DATETIME']
    df_time_test = df_trn2['DATETIME']
    df_trn2 = df_trn2.replace(-999, 0)
    labels = df_test['ATT_FLAG']
    del df_trn1['DATETIME']
    del df_trn2['DATETIME']
    del df_test['DATETIME']
    del df_trn1['ATT_FLAG']
    del df_test['ATT_FLAG']

    #normalising
    scaler = StandardScaler()
    df_trn1 = scaler.fit_transform(df_trn1)
    df_trn2 = scaler.fit_transform(df_trn2)
    df_test = scaler.fit_transform(df_test)

    ##################
    pca = decomposition.PCA(n_components=43)#43
    pca.fit(df_trn1)
    pca_model = pca.transform(df_trn1)


    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_

    # Matrix P represents principal components corresponding to normal subspace
    P = np.transpose(eigenvectors[:-32])
    P_T = np.transpose(P)
    C = np.dot(P, P_T) #linear operator

    C2_y = np.dot(df_test, (C)) #(I-PP^T)*y
    SPE = np.square(df_test - C2_y).sum(axis=1)

    threshold = 6.5

    predictions = np.zeros((df_test.shape[0]))
    for i in range(len(predictions)):
        if (SPE[i] > threshold):
            predictions[i] = 1

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(df_test.shape[0]):
        if (labels[i] == 1 and predictions[i] == 1):
            tp = tp + 1
        if (labels[i] == 0 and predictions[i] == 1):
            fp = fp + 1
        if (labels[i] == 0 and predictions[i] == 0):
            tn = tn + 1
        if (labels[i] == 1 and predictions[i] == 0):
            fn = fn + 1
    print 'tp: ', tp
    print 'fp: ', fp
    print 'fn: ', fn
    print 'tn: ', tn
    print 'precision:', 1.0 * tp / (tp + fp)
    print 'recall:', 1.0 * tp / (tp + fn)


    figure, ax = plt.subplots(figsize=[15,10])
    x = df_time2
    y = SPE
    plt.xlabel('Time')
    plt.ylabel('Residual Vector')
    ax.hlines(y=threshold, xmin=df_time2[0], xmax=df_time2.iloc[-1], linewidth=2, color='r')
    ax.plot(x, y)
    plt.show()


    ######################


    #COMBINING PCA with NGRAMS for potential bonus


    ####################
    columns = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'F_PU2', 'F_PU3', 'F_PU4', 'F_PU5', 'F_PU6','F_PU7','F_PU8', 'F_PU9','F_PU10', 'F_PU11', 'F_V2', 'S_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415', 'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']
    #predictions = np.zeros((df_test.shape[0]))
    n = 8 #max n for ngram
    s = 3 #number of letters
    df_trn1, df_trn2, df_test = utils.import_datasets()
    for col in columns:
        df_trn1[col] = list(ts_to_string(df_trn1[col], cuts_for_asize(s)))
        df_test[col] = list(ts_to_string(df_test[col], cuts_for_asize(s)))

        ngrams = discrete.find_ngrams(df_trn1[col] , n)

        occurances =  discrete.find_occurances(ngrams)
        probas = discrete.find_probabilities(occurances)

        ngramsTest = discrete.find_ngrams(df_test[col], n)


        counter = n-1
        for r in ngramsTest[n-2]:
            if probas[n-2].get(r) < 0.000000001:
                predictions[counter] = 1
            counter += 1

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(df_test.shape[0]):
        if (labels[i] == 1 and predictions[i] == 1):
            tp = tp + 1
        if (labels[i] == 0 and predictions[i] == 1):
            fp = fp + 1
        if (labels[i] == 0 and predictions[i] == 0):
            tn = tn + 1
        if (labels[i] == 1 and predictions[i] == 0):
            fn = fn + 1

    print ''
    print ''
    print "PCA with NGRAMS combined:"
    print 'tp: ', tp
    print 'fp: ', fp
    print 'fn: ', fn
    print 'tn: ', tn
    print 'precision:', 1.0 * tp / (tp + fp)
    print 'recall:', 1.0 * tp / (tp + fn)