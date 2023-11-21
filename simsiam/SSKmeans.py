#!/usr/bin/env python

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn import datasets

digists = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digists.data, digists.target, test_size=0.5)

X_train0, X_train1, y_train0, _ = train_test_split(X_train, y_train, test_size=0.95)


class SupervisedKMeans(ClassifierMixin, KMeans):
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.centers_ = np.array([np.mean(X[y == c], axis=0) for c in self.classes])
        self.cluster_centers_ = self.centers_
        return self

    def predict(self, X):
        ed = euclidean_distances(X, self.cluster_centers_)
        return [self.classes[k] for k in np.argmin(ed, axis=1)]

    def score(self, X, y):
        y_ = self.predict(X)
        return np.mean(y == y_)


class SemiKMeans(SupervisedKMeans):
    def fit(self, X0, y0, X1):
        """To fit the semisupervised model

        Args:
            X0 (array): input variables with labels
            y0 (array): labels
            X1 (array): input variables without labels

        Returns:
            the model
        """
        classes0 = np.unique(y0)
        classes1 = np.setdiff1d(np.arange(self.n_clusters), classes0)
        self.classes = np.concatenate((classes0, classes1))

        X = np.row_stack((X0, X1))
        n1 = len(classes1)
        mu0 = SupervisedKMeans().fit(X0, y0).centers_
        if n1:
            centers, indices = kmeans_plusplus(X1, n_clusters=n1)
            self.cluster_centers_ = np.row_stack((centers, mu0))
        else:
            self.cluster_centers_ = mu0
        for _ in range(30):
            ED = euclidean_distances(X1, self.cluster_centers_)
            y1 = [self.classes[k] for k in np.argmin(ED, axis=1)]
            y = np.concatenate((y0, y1))
            self.cluster_centers_ = np.array([np.mean(X[y == c], axis=0) for c in self.classes])
        return self


if __name__ == '__main__':
    km = SemiKMeans(n_clusters=10)
    km.fit(X_train0, y_train0, X_train1)  # y_test0 is unknown
    skm = SupervisedKMeans(n_clusters=10)
    skm.fit(X_train0, y_train0)
    print(f"""
    # clusters: 10
    # samples: {X_train0.shape[0]} + {X_train1.shape[0]}

    SemiKMeans: {km.score(X_test, y_test)}
    SupervisedKMeans: {skm.score(X_test, y_test)}
    """)

