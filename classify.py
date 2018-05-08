import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
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
del df['txid'] #otherwise 10% accuracy with tress/randomforest

del df['bin']
del df['accountcode']
del df['mail_id']
del df['ip_id']
del df['card_id']

del df['issuercountrycode']
del df['shoppercountrycode']


print "classifier train features:", list(df)
if __name__ == "__main__":
    #splitting data in x and y
    y = df[['simple_journal']].copy()
    del df['simple_journal']
    y = y.simple_journal.values
    x =  df #df[['txvariantcode', 'cardverificationcodesupplied']]
    x = x.as_matrix()

    from sklearn.utils import shuffle
    x, y = shuffle(x, y)

    #data splitting into a train set and a test set
    skf = StratifiedKFold(n_splits = 10) #stratifiedKfold desging choice!
    skf.split(x, y)

    for train_index, test_index in skf.split(x,y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        oversampler = SMOTE(ratio='minority', random_state=None, k_neighbors=3, m=None, m_neighbors=50, out_step=0.2, kind='regular', svm_estimator=None, n_jobs=1)
        x_train, y_train = oversampler.fit_sample(x_train, y_train)


        # clf1 = RandomForestClassifier(n_estimators=400, oob_score =False, n_jobs=-1).fit(x_train, y_train)
        # prediction_probability1 = clf1.predict_proba(x_test)[:,1]
        # # print 'clf1 ready'
        # clf2 = LogisticRegression(penalty='l1', C=400.0, class_weight={0:9}).fit(x_train, y_train)#penalty='l2', dual=False, tol=0.0001, C=400.0, fit_intercept=True, intercept_scaling=1, class_weight={0:9}, random_state=None, solver='saga', max_iter=200, multi_class='ovr', verbose=0, warm_start=False, n_jobs=-1).fit(x_train, y_train)
        # prediction_probability2 = clf2.predict_proba(x_test)[:,1]
        # print 'clf2 ready'
        # clf3 = KNeighborsClassifier(n_neighbors=2, leaf_size=200, p=2, weights='uniform', n_jobs=-1).fit(x_train, y_train)
        # prediction_probability3 = clf3.predict_proba(x_test)[:,1]
        # print 'clf3 ready'
        # clf4 = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(15,), random_state=0).fit(x_train, y_train)
        # prediction_probability4 = clf4.predict_proba(x_test)[:,1]

        # clf5 = AdaBoostClassifier(n_estimators=100, base_estimator=clf1).fit(x_train, y_train)
        # prediction_probability5 = clf5.predict_proba(x_test)[:,1]



        clf1 = DecisionTreeClassifier(min_samples_split=50)
        clf2 = RandomForestClassifier()
        clf3 = MLPClassifier(hidden_layer_sizes=(15,), alpha=0.01)
        clf4 = LogisticRegression(penalty='l1', C=400.0, class_weight={0:9})
        clf5 = KNeighborsClassifier(n_neighbors=2, leaf_size=200, p=2, weights='uniform', n_jobs=-1)

        clfV = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2), ('mlp', clf3), ('lr', clf4), ('kn', clf5)], n_jobs=-1)
        clfV.fit(x_train, y_train)
        prediction_probability = clfV.predict(x_test)



        # predictions = np.vstack((prediction_probability1, prediction_probability2, prediction_probability3))
        # predictions = predictions.mean(0)
        # predictions = np.rint(np.array(predictions))

        tn, fp, fn, tp = metrics.confusion_matrix(y_test, np.rint(np.array(prediction_probability))).ravel()
        accuracy = float(tp + tn) / (tp + tn + fn + fp)
        recall = float(tp) / (tp + fn)
        precision = float(tp) / (tp + fp)

        print ''
        print 'tn:', tn, ' fp:', fp, ' fn:', fn, ' tp:', tp, ' accuracy:', accuracy, ' recall:', recall, 'precision', precision


