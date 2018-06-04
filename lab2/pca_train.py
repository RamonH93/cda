import utils
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df_trn1, df_trn2, df_test = utils.import_datasets()
    df_time1 = df_trn1['DATETIME']
    # df_time2 = df_trn2['DATETIME']
    df_time_test = df_trn2['DATETIME']
    # df_trn2 = df_trn2.replace(-999, 0)
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
    # C2_y = np.dot(df_trn2, (np.identity(43)-C)) #(I-PP^T)*y
    # SPE = np.square(df_trn2 - C2_y).sum(axis=1)

    x_axis = np.arange(1, 44, 1)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Captured')
    plt.plot(x_axis, pca.explained_variance_ratio_)
    plt.show()

    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Variance Captured')
    plt.plot(x_axis, pca.explained_variance_ratio_.cumsum())
    plt.show()

    #calculating h0 value and then the threshold
    lambda1 = eigenvalues
    lambda2 = np.power(eigenvalues, 2)
    lambda3 = np.power(eigenvalues, 3)
    phi1 = sum(lambda1[15:])
    phi2 = sum(lambda2[15:])
    phi3 = sum(lambda3[15:])
    h_0 = 1 - ((2.0 * phi1 * phi3) / (3 * (phi2 ** 2)))
    Ca = 3
    threshold = phi1 * np.power(1.0 * (Ca * np.sqrt(2 * phi2 * (h_0 ** 2)) / phi1) + 1 + (1.0 * (phi2 * h_0 * (h_0 - 1)) / (phi1 ** 2)), (1.0 / h_0))
    print phi1, phi2, phi3, h_0, threshold

    #threshold = 100


    for i in range(len(df_trn1)):
        if (pca_residuals[i] > threshold):
            print 'Row ', i, 'contains an abnormality and should be removed'


    figure, ax = plt.subplots(figsize=[15,10])
    x = df_time1
    y = pca_residuals
    plt.xlabel('Time')
    plt.ylabel('Residual Vector')
    ax.hlines(y=threshold, xmin=df_time1[0], xmax=df_time1.iloc[-1], linewidth=2, color='r')
    ax.plot(x, y)
    plt.show()



