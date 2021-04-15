from sklearn.feature_selection import *


class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type, scoring):
        '''Custom univariate feature selection wrapper on
        different univariate feature selection models from
        scikit-learn.
        :param n_features:SelectPercentile if float else SelectKbase
        :param problem_type:Classification or regression
        :param scoring:Scoring function ,string
        '''
        if problem_type == 'classification':
            valid_scoring = {'f_classif': f_classif,
                             'chi2': chi2,
                             'mutual_info_classif': mutual_info_classif}
        else:
            valid_scoring = {'f_regression': f_regression,
                             'mutual_info_regression': mutual_info_regression}

        if scoring not in valid_scoring:
            raise Exception("Invalid scoring function")

        if isinstance(n_features, int):
            self.selection = SelectKBest(valid_scoring[scoring], k=n_features)

        elif isinstance(n_features, float):
            self.selection = SelectPercentile(valid_scoring[scoring], percentile=int(n_features * 100))

        else:
            raise Exception("Invalid type of feature")


def fit(self, X, y):
    return self.selection.fit(X, y)


def transform(self, X):
    return self.selection.transform(X)


def fit_transform(self, X, y):
    return self.selection.fit_transform(X, y)
