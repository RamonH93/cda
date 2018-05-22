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
from sklearn.feature_selection import RFECV
from sklearn.utils import shuffle

# imports for the RFECV override
from sklearn.feature_selection import rfe
from sklearn.utils import check_X_y
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _safe_split, _score
from sklearn.metrics.scorer import check_scoring
from sklearn.feature_selection import RFE

# override rfe method to SMOTE train samples
def _rfe_single_fit(rfe, estimator, X, y, train, test, scorer):
    """
    Return the score for a fit across one fold.
    """
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    oversampler = SMOTE(ratio='minority', random_state=None, k=None, k_neighbors=5, m=None, m_neighbors=10, out_step=0.5, kind='regular', svm_estimator=None, n_jobs=1)
    X_train, y_train = oversampler.fit_sample(X_train, y_train)

    return rfe._fit(
        X_train, y_train, lambda estimator, features:
        _score(estimator, X_test[:, features], y_test, scorer)).scores_

# override RFECV class to use overridden _rfe_single_fit
class SMOTE_RFECV(RFECV):
    def __init__(self, estimator, step=1, cv=None, scoring=None, verbose=0,
                     n_jobs=1):
        super(SMOTE_RFECV, self).__init__(estimator=estimator, step=step, cv=cv, scoring=scoring, verbose=verbose,
                     n_jobs=n_jobs)

    def fit(self, X, y):
        """Fit the RFE model and automatically tune the number of selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the total number of features.

        y : array-like, shape = [n_samples]
            Target values (integers for classification, real numbers for
            regression).
        """
        X, y = check_X_y(X, y, "csr")

        # Initialization
        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]
        n_features_to_select = 1

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        rfe = RFE(estimator=self.estimator,
                                            n_features_to_select=n_features_to_select,
                                            step=self.step, verbose=self.verbose)

        # Determine the number of subsets of features by fitting across
        # the train folds and choosing the "features_to_select" parameter
        # that gives the least averaged error across all folds.

        # Note that joblib raises a non-picklable error for bound methods
        # even if n_jobs is set to 1 with the default multiprocessing
        # backend.
        # This branching is done so that to
        # make sure that user code that sets n_jobs to 1
        # and provides bound methods as scorers is not broken with the
        # addition of n_jobs parameter in version 0.18.

        if self.n_jobs == 1:
            parallel, func = list, _rfe_single_fit
        else:
            parallel, func, = Parallel(n_jobs=self.n_jobs), delayed(_rfe_single_fit)

        scores = parallel(
            func(rfe, self.estimator, X, y, train, test, scorer)
            for train, test in cv.split(X, y))

        scores = np.sum(scores, axis=0)
        n_features_to_select = max(
            n_features - (np.argmax(scores) * step),
            n_features_to_select)

        # Re-execute an elimination with best_k over the whole set
        rfe = RFE(estimator=self.estimator,
                                            n_features_to_select=n_features_to_select, step=self.step)

        rfe.fit(X, y)

        # Set final attributes
        self.support_ = rfe.support_
        self.n_features_ = rfe.n_features_
        self.ranking_ = rfe.ranking_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.transform(X), y)

        # Fixing a normalization error, n is equal to get_n_splits(X, y) - 1
        # here, the scores are normalized by get_n_splits(X, y)
        self.grid_scores_ = scores[::-1] / cv.get_n_splits(X, y)
        return self


if __name__ == '__main__':

    #set pandas output width, for printing dataframes and debugging purposes
    desired_width = 320
    pd.set_option('display.width', desired_width)

    df = pd.read_csv('clean.csv')
    # df = df.head(10000)

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

    x, y = shuffle(x, y)
    # oversampler = SMOTE(ratio='minority', random_state=None, k=None, k_neighbors=5, m=None, m_neighbors=10, out_step=0.5, kind='regular', svm_estimator=None, n_jobs=1)
    # x, y = oversampler.fit_sample(x, y)

    rf = RandomForestClassifier(n_estimators=400, n_jobs=1)
    lg = LogisticRegression()

    rfecv = SMOTE_RFECV(estimator=rf, step=1, cv=10, scoring='precision', verbose=1, n_jobs=1)
    rfecv.fit(x, y)

    print rfecv.support_
    print rfecv.ranking_

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.ylim(ymin=0, ymax=1)
    plt.show()
