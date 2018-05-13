import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


#set pandas output width
desired_width = 320
pd.set_option('display.width', desired_width)

df = pd.read_csv('clean.csv')

print "The number of fraudulent transaction very much smaller compared to the safe transactions:"
print len(df[df["simple_journal"]==1]), "fraud cases and ", len(df[df["simple_journal"]==0]), "safe cases"

#removing the dates, as SMOTE cannot handle those, also txid removed
del df['bookingdate']
del df['creationdate']
del df['txid']


print "classifier train features:", list(df)

#splitting data in x and y
y = df[['simple_journal']].copy()
del df['simple_journal']
y = y.simple_journal.values
x = df

#data splitting into a train set and a test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

#This is the oversampler
oversampler = SMOTE(ratio='minority')
x_train_SMOTE,y_train_SMOTE=oversampler.fit_sample(x_train,y_train)

def getClassifierScores(clf):
    prediction_probability = clf.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, prediction_probability)
    fpr, tpr, thresholds = roc_curve(y_test, prediction_probability)
    return auc, fpr, tpr

def plotROCcurves(fpr, tpr, fpr_SMOTE, tpr_SMOTE, auc, auc_SMOTE, title):
    plt.title(title)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, color='green', label='No SMOTE = %0.2f' % auc)
    plt.plot(fpr_SMOTE, tpr_SMOTE, color = 'orange', label='SMOTE = %0.2f' % auc_SMOTE)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()


#logistic classifiers  - good
clf_logistic_SMOTE=LogisticRegression().fit(x_train_SMOTE,y_train_SMOTE)
clf_logistic=LogisticRegression().fit(x_train,y_train)

auc, fpr, tpr = getClassifierScores(clf_logistic)
auc_SMOTE, fpr_SMOTE, tpr_SMOTE = getClassifierScores(clf_logistic_SMOTE)
plotROCcurves(fpr, tpr, fpr_SMOTE, tpr_SMOTE, auc, auc_SMOTE, "ROC Logistic Regression")


#random forest classifiers - good
clf_randomForest_SMOTE = RandomForestClassifier(n_estimators=50, criterion='gini')
clf_randomForest_SMOTE.fit(x_train_SMOTE,y_train_SMOTE)
clf_randomForest = RandomForestClassifier(n_estimators=50, criterion='gini')
clf_randomForest.fit(x_train,y_train)

auc, fpr, tpr = getClassifierScores(clf_randomForest)
auc_SMOTE, fpr_SMOTE, tpr_SMOTE = getClassifierScores(clf_randomForest_SMOTE)
plotROCcurves(fpr,tpr,fpr_SMOTE,tpr_SMOTE, auc, auc_SMOTE, "ROC Random Forest")

# #neural network, multilayer perceptron
clf_MLP_SMOTE = MLPClassifier()
clf_MLP_SMOTE.fit(x_train_SMOTE,y_train_SMOTE)
clf_MLP = MLPClassifier()
clf_MLP.fit(x_train,y_train)

auc, fpr, tpr = getClassifierScores(clf_MLP)
auc_SMOTE, fpr_SMOTE, tpr_SMOTE = getClassifierScores(clf_MLP_SMOTE)
plotROCcurves(fpr,tpr,fpr_SMOTE,tpr_SMOTE, auc, auc_SMOTE, clf_MLP_SMOTE)


#KneighborsClassifier classifiers
clf_KNeigbors_SMOTE = KNeighborsClassifier(30)
clf_KNeigbors_SMOTE.fit(x_train_SMOTE,y_train_SMOTE)
clf_KNeigbors = KNeighborsClassifier(3)
clf_KNeigbors.fit(x_train,y_train)

auc, fpr, tpr = getClassifierScores(clf_KNeigbors)
auc_SMOTE, fpr_SMOTE, tpr_SMOTE = getClassifierScores(clf_KNeigbors_SMOTE)
plotROCcurves(fpr,tpr,fpr_SMOTE,tpr_SMOTE, auc, auc_SMOTE, "ROC SVC")

