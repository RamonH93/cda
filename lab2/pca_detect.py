import utils
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df_trn1, df_trn2, df_test = utils.import_datasets()
    #df_trn1 = df_trn1.drop(df_trn1.index[[2348, 2349, 2350, 2854, 2855, 3000, 3001, 3023, 3024, 3336, 3337, 3356, 3357, 3358, 3861, 3862, 4365, 4366, 7222, 7223]])
    #df_trn1 = df_trn1.drop(df_trn1.index[[1764,2461,3000,3001,3336,3337,5572,6084,7573,8646]])
    df_time1 = df_trn1['DATETIME']
    df_time2 = df_trn2['DATETIME']
    df_time_test = df_trn2['DATETIME']
    df_trn2 = df_trn2.replace(-999, 0)
    labels = df_trn2['ATT_FLAG']

    del df_trn1['DATETIME']
    del df_trn2['DATETIME']
    del df_test['DATETIME']
    del df_trn1['ATT_FLAG']
    del df_trn2['ATT_FLAG']

    #normalising
    scaler = StandardScaler()
    df_trn1 = scaler.fit_transform(df_trn1)
    df_trn2 = scaler.fit_transform(df_trn2)
    df_test = scaler.fit_transform(df_test)

    ##################
    pca = decomposition.PCA(n_components=43)#36 #new:43
    pca.fit(df_trn1)
    pca_model = pca.transform(df_trn1)


    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_

    # Matrix P represents principal components corresponding to normal subspace
    P = np.transpose(eigenvectors[:-28])#26
    P_T = np.transpose(P)
    C = np.dot(P, P_T) #linear operator

    C_y = np.dot(df_trn1, C)
    pca_residuals = np.square(df_trn1 - C_y).sum(axis=1) #squared prediction error
    C2_y = np.dot(df_trn2, (np.identity(43)-C)) #(I-PP^T)*y
    SPE = np.square(df_trn2 - C2_y).sum(axis=1)

    threshold = 75

    predictions = np.zeros((df_trn2.shape[0]))
    for i in range(len(predictions)):
        if (SPE[i] > threshold):
            predictions[i] = 1

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(df_trn2.shape[0]):
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
