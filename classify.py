import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#set pandas output width, for printing dataframes and debugging purposes
desired_width = 500
pd.set_option('display.width', desired_width)

df = pd.read_csv('clean1.csv')

print "The number of fraudulent transaction very much smaller compared to the safe transactions:"
print len(df[df["simple_journal"]==1]), "fraud cases and ", len(df[df["simple_journal"]==0]), "safe cases"

#removing the dates, as SMOTE cannot handle those, also txid removed
del df['bookingdate']
del df['creationdate']
del df['txid'] #otherwise 100% accuracy with tress/randomforest
del df['shoppercountrycode']
del df['bin']
del df['card_id']
del df['accountcode']
del df['mail_id']
del df['ip_id']
del df['issuercountrycode']

# del df['binnumberofcountries']
# del df['bintotalamount']
# del df['minuteamount']


# function that returns an array with binary data based on treshold t
def adjusted_classes(y_scores, t):
    return [1 if y >= t else 0 for y in y_scores]


if __name__ == "__main__":
    #splitting data in x and y
    y = df[['simple_journal']].copy()
    del df['simple_journal']
    y = y.simple_journal.values
    x =  df #df[['txvariantcode', 'cardverificationcodesupplied']]
    x = x.as_matrix()
    x, y = shuffle(x, y)

    #data splitting into a train set and a test set
    skf = StratifiedKFold(n_splits = 10) #stratifiedKfold desging choice!

    tn = np.zeros(10)
    fp = np.zeros(10)
    fn = np.zeros(10)
    tp = np.zeros(10)
    i = 0
    print df.head
    for train_index, test_index in skf.split(x,y):

        #print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #applying SMOTE
        oversampler = SMOTE(ratio=0.01, k_neighbors=3)#ratio=0.01, k_neighbors=3)#ratio='minority', random_state=None, k_neighbors=3, m=None, m_neighbors=50, out_step=0.2, kind='regular', svm_estimator=None, n_jobs=-1)
        x_train, y_train = oversampler.fit_sample(x_train, y_train)

        #only normalizing for the MLPclassifier
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_trainN = scaler.transform(x_train)
        x_testN = scaler.transform(x_test)

        print ''
        print 'Preprocessing fold', i, 'done.'

        # clf1 = DecisionTreeClassifier(min_samples_split=50)
        clf2 = RandomForestClassifier(n_estimators=600, n_jobs=-1)
        clf3 = MLPClassifier(solver='sgd', max_iter=300, alpha=1e-6, early_stopping=False)#alpha=0.0001, random_state=0, activation='tanh')#hidden_layer_sizes=(5, 2)
        #clf4 = LogisticRegression(penalty='l1', C=400.0)
        clf5 = KNeighborsClassifier(n_neighbors=3, leaf_size=200, p=2, weights='distance', n_jobs=-1)
        #clf6 = GaussianNB()
        clfV = VotingClassifier(estimators=[('rf', clf2), ('lr', clf3), ('kn', clf5)], n_jobs=-1, voting='soft')

        clf2.fit(x_train, y_train)
        clf3.fit(x_trainN, y_train)

        predictions2 = clf2.predict_proba(x_test)[:,1]
        predictions3 = clf3.predict_proba(x_testN)[:,1]

        predictions = np.vstack((predictions2, predictions3))
        predictions = predictions.mean(0)
        predictions = adjusted_classes(predictions, 0.15)


        tn[i], fp[i], fn[i], tp[i] = metrics.confusion_matrix(y_test, predictions).ravel()
        accuracy = float(tp[i] + tn[i]) / (tp[i] + tn[i] + fn[i] + fp[i])
        recall = float(tp[i]) / (tp[i] + fn[i])
        precision = float(tp[i]) / (tp[i] + fp[i])

        print 'tn:', tn[i], ' fp:', fp[i], ' fn:', fn[i], ' tp:', tp[i], ' accuracy:', accuracy, ' recall:', recall, 'precision', precision
        i = i+1

    print 'tn:', sum(tn), ' fp:', sum(fp), ' fn:', sum(fn), ' tp:', sum(tp)
