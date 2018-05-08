import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#set pandas output width, for printing dataframes and debugging purposes
desired_width = 320
pd.set_option('display.width', desired_width)

df = pd.read_csv('clean.csv')

print "The number of fraudulent transaction very much smaller compared to the safe transactions:"
print len(df[df["simple_journal"]==1]), "fraud cases and ", len(df[df["simple_journal"]==0]), "safe cases"

#removing the dates, as SMOTE cannot handle those, also txid removed
del df['bookingdate']
del df['creationdate']
del df['txid']

#splitting data in x and y
y = df[['simple_journal']].copy()
del df['simple_journal']
y = y.simple_journal.values
x =  df #df[['txvariantcode', 'cardverificationcodesupplied']]
x = x.as_matrix()

from sklearn.utils import shuffle
x, y = shuffle(x, y)

#data splitting into a train set and a test set
skf = StratifiedKFold(n_splits = 10)
skf.split(x, y)

for train_index, test_index in skf.split(x,y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    oversampler = SMOTE(ratio='minority', random_state=None, k=None, k_neighbors=5, m=None, m_neighbors=10, out_step=0.5, kind='regular', svm_estimator=None, n_jobs=1)
    x_train, y_train = oversampler.fit_sample(x_train, y_train)

    clf1 = RandomForestClassifier(n_estimators=400, n_jobs=10).fit(x_train, y_train)
    prediction_probability1 = clf1.predict_proba(x_test)[:,1]
    # clf2 = LogisticRegression().fit(x_train, y_train)
    # prediction_probability2 = clf2.predict_proba(x_test)[:,1]
    # clf3 = KNeighborsClassifier(30).fit(x_train, y_train)
    # prediction_probability3 = clf3.predict_proba(x_test)[:,1]
    #
    #
    #
    # predictions = np.vstack((prediction_probability1, prediction_probability2, prediction_probability3))
    # predictions = predictions.mean(0)
    # predictions = np.rint(np.array(predictions))

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, np.rint(np.array(prediction_probability1))).ravel()
    accuracy = float(tp + tn) / (tp + tn + fn + fp)
    recall = float(tp) / (tp + fn)
    precision = float(tp) / (tp + fp)

    print ''
    print 'tn:', tn, ' fp:', fp, ' fn:', fn, ' tp:', tp, ' accuracy:', accuracy, ' recall:', recall, 'precision', accuracy





# #This is the oversampler
# oversampler = SMOTE(kind='regular')
# x_train,y_train=oversampler.fit_sample(X_train,y_train)
#
# clf=LogisticRegression().fit(x_train,y_train)
#
# predictions = clf.predict(x_test)
# tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
# print tn, fp, fn, tp