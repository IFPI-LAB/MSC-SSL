import argparse
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import scipy.spatial
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from SSKmeans import SemiKMeans

import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
# args
parser = argparse.ArgumentParser(description='PyTorch Infer')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num_classes', default=3, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--load_npz', default='feature.npz', type=str,
                    help='data path')

colors = ['navy', 'turquoise', 'darkorange']


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.1)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


def tobacco_cluster_acc(cluster_label, ori_gt):
    labels = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
    best = -1
    for item in labels:
        acc, new_cluster_label = relabel_cluster_acc(ori_gt, cluster_label, item)
        if acc > best:
            best = acc
            re_cluster_label = new_cluster_label
    return best, re_cluster_label


def relabel_cluster_acc(ori_gt, cluster_label, new):
    new_cluster_label = np.ones_like(cluster_label) * -1
    new_cluster_label[np.where(cluster_label == 0)] = new[0]
    new_cluster_label[np.where(cluster_label == 1)] = new[1]
    new_cluster_label[np.where(cluster_label == 2)] = new[2]
    acc = np.mean(new_cluster_label.ravel() == ori_gt.ravel()) * 100

    return acc, new_cluster_label


def relabel_gt_acc(ori_gt, cluster_label, new):
    new_gt = np.ones_like(ori_gt) * -1
    new_gt[np.where(ori_gt == 0)] = new[0]
    new_gt[np.where(ori_gt == 1)] = new[1]
    new_gt[np.where(ori_gt == 2)] = new[2]
    acc = np.mean(cluster_label.ravel() == new_gt.ravel()) * 100

    return acc, new_gt


def plot_gmm(estimator, X_train, y_train, save_fig=None):
    ###############################################################
    ####  画图
    ###############################################################
    centers = estimator.means_
    # 预测标签
    y_train_pred = estimator.predict(X_train)
    train_accuracy, re_y_train_pred = tobacco_cluster_acc(y_train_pred, y_train)
    # print('train acc:{:.6f}'.format(train_accuracy))

    ## 真实标签
    fig = plt.figure(figsize=(15,4))
    fig_1 = plt.subplot(1, 3, 1)
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=label_all, s=1, marker=".")
    legend1 = fig_1.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    fig_1.add_artist(legend1)
    plt.title("GT label")
    # plt.show()
    ## 预测标签
    # plt.figure(2)
    fig_2 = plt.subplot(1, 3, 2)
    make_ellipses(estimator, fig_2)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=re_y_train_pred, s=1, marker=".")
    legend1 = fig_2.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    fig_2.add_artist(legend1)
    plt.scatter(centers[:, 0], centers[:, 1], c='r', s=30, marker="*")
    plt.title("Pred label")
    # plt.show()

    xlim = [np.min(X_train[:, 0]), np.max(X_train[:, 0])]
    ylim = [np.min(X_train[:, 1]), np.max(X_train[:, 1])]
    dinter = (np.maximum(xlim[1], ylim[1]) - np.minimum(xlim[0], ylim[0])) / 100.0
    x = np.arange(int(xlim[0]), int(xlim[1]), int(dinter))
    y = np.arange(int(ylim[0]), int(ylim[1]), int(dinter))
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -estimator.score_samples(XX)
    Z = Z.reshape(X.shape)
    # CS = plt.contour(
    #     X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    # )
    # plt.figure(3)
    fig_3 = plt.subplot(1, 3, 3)
    CS = plt.contour(X, Y, Z, 30, linewidths=1)
    CB = plt.colorbar(CS, shrink=0.8, extend="both")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=re_y_train_pred, s=1, marker=".")
    plt.scatter(centers[:, 0], centers[:, 1], c='r', s=30, marker="*")

    plt.title("Negative log-likelihood predicted by a GMM")
    plt.axis("tight")
    # plt.show()
    if save_fig is not None:
        plt.savefig(save_fig)


if __name__ == "__main__":
    args = parser.parse_args()

    # load embedding
    data = np.load(os.path.join(args.load_npz))
    embedding_all = data['embedding_all']
    label_all = data['label_all']
    im_path_all = data['im_path_all'].tolist()
    cos_similarity = data['cos_similarity']

    X_train, y_train = embedding_all, label_all[:, 0]

    # Data Normalization
    scale = StandardScaler()  # Standard Deviation Normalization
    X_train = scale.fit_transform(X_train)

    # Dimensionality Reduction
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)

    picked_label = np.ones_like(y_train) * -1
    picked_label_kmeans = np.ones_like(y_train) * -1
    picked_num_kmeans = np.zeros((args.num_classes))
    picked_num = np.zeros((args.num_classes))
    picked_dick = {}
    for iter in range(140):
        print('iter {}:'.format(iter))
        fig_path = os.path.join('figures', 'iter_{}.png'.format(iter))
        if iter == 0:
            # cluster: GMM
            estimator = GaussianMixture(n_components=args.num_classes, covariance_type='full', random_state=args.seed)
            estimator.fit(X_train)  # Train the other parameters using the EM algorithm.
            plot_gmm(estimator, X_train, y_train, fig_path)
        else:
            # semi-GMM
            X_l = X_train[picked_label != -1]
            y_l = y_train[picked_label != -1]
            X_u = X_train
            y_u = y_train
            km = SemiKMeans(n_clusters=args.num_classes)
            km.fit(X_l, y_l, X_u)
            means_init = km.cluster_centers_
            weights_init = np.array([1 / 3, 1 / 3, 1 / 3])
            estimator = GaussianMixture(n_components=args.num_classes, covariance_type='full',
                                             means_init=means_init, weights_init=weights_init, random_state=args.seed)
            estimator.fit(X_train)
            plot_gmm(estimator, X_train, y_train, fig_path)

        # Predict Labels
        y_train_pred = estimator.predict(X_train)
        train_accuracy, re_y_train_pred = tobacco_cluster_acc(y_train_pred, y_train)
        print('train acc:{:.6f}'.format(train_accuracy))

        # Predict Probability
        y_train_pred_prob = estimator.predict_proba(X_train)
        y_train_pred_prob_max = np.max(y_train_pred_prob, axis=1)

        # Distance of Sample to the Center
        centers = estimator.means_
        re_centers = np.zeros_like(centers)
        for i in range(args.num_classes):
            sample_center = np.mean(X_train[re_y_train_pred == i], axis=0).reshape((1, 2))
            dist = scipy.spatial.distance.cdist(centers, sample_center, metric='euclidean')
            re_centers[i] = centers[np.argmin(dist)]
        dists_all = scipy.spatial.distance.cdist(re_centers, X_train, metric='euclidean').T
        dists = np.choose(re_y_train_pred, dists_all.T)
        print('centers: {}, {}, {}'.format(re_centers[0], re_centers[1], re_centers[2]))

        # Sample Selection:
        # Select samples based on two strategies,
        # 1. Choose Samples Close to the Center,
        # 2. Choose Samples with the Lowest Probability.
        # Both strategies should ensure that at least one sample from each category is selected.

        # Selection Strategy 1: Choose Samples Close to the Center
        picked_num_iter = np.zeros((args.num_classes))
        cos_thre = 0.97
        for i in range(args.num_classes):
            temp_num = 0
            try:
                while temp_num == 0:
                    valid_index = np.where((re_y_train_pred==i) & (picked_label_kmeans == -1))[0]
                    min_dist_idx = valid_index[dists[valid_index].argmin()]
                    name = '-'.join([im_path_all[min_dist_idx][0].split('\\')[-2], im_path_all[min_dist_idx][0].split('\\')[-1]])
                    name = '-'.join(name.split('-')[0:-1])
                    picked_label_kmeans[min_dist_idx] = y_train[min_dist_idx]
                    picked_num_kmeans[int(y_train[min_dist_idx])] += 1
                    picked_cos = cos_similarity[min_dist_idx, picked_label != -1]
                    if iter == 0 or np.max(picked_cos) < cos_thre:
                        picked_dick[name] = 1
                        picked_label[min_dist_idx] = y_train[min_dist_idx]
                        picked_num[int(y_train[min_dist_idx])] += 1
                        picked_num_iter[int(y_train[min_dist_idx])] += 1
                        # print('picked num: {}, pred label: {}, gt label: {}, im path: {}'.format(picked_num, i, y_train[min_dist_idx], im_path_all[min_dist_idx]))
                        if y_train[min_dist_idx] == i:
                            temp_num += 1
            except:
                continue
        print("pick dist min: unripe={}, ripe={}, overripe={}".format(picked_num_iter[0], picked_num_iter[1], picked_num_iter[2]))

        # Selection Strategy 2: Choose Samples with the Lowest Probability
        picked_num_iter = np.zeros((args.num_classes))
        for i in range(args.num_classes):
            temp_num = 0
            try:
                while temp_num == 0:
                    valid_index = np.where((re_y_train_pred == i) & (picked_label_kmeans == -1))[0]
                    min_prob_idx = valid_index[y_train_pred_prob_max[valid_index].argmin()]
                    name = '-'.join([im_path_all[min_prob_idx][0].split('\\')[-2], im_path_all[min_prob_idx][0].split('\\')[-1]])
                    name = '-'.join(name.split('-')[0:-1])
                    picked_label_kmeans[min_prob_idx] = y_train[min_prob_idx]
                    picked_num_kmeans[int(y_train[min_prob_idx])] += 1
                    picked_cos = cos_similarity[min_prob_idx, picked_label != -1]
                    if iter == 0 or np.max(picked_cos) < cos_thre:
                        picked_dick[name] = 1
                        picked_label[min_prob_idx] = y_train[min_prob_idx]
                        picked_num[int(y_train[min_prob_idx])] += 1
                        picked_num_iter[int(y_train[min_prob_idx])] += 1
                        # print('picked num: {}, pred label: {}, gt label: {}, im path: {}'.format(picked_num, i, y_train[min_prob_idx], im_path_all[min_prob_idx]))
                        if y_train[min_prob_idx] == i:
                            temp_num += 1
            except:
                continue

        print("pick prob min: unripe={}, ripe={}, overripe={}".format(
            picked_num_iter[0], picked_num_iter[1],picked_num_iter[2]))

        print('total picked num={}, unripe={}, ripe={}, overripe={}, rate={:.2f}%'.format(
            np.sum(picked_num), picked_num[0], picked_num[1], picked_num[2], np.sum(picked_num)/y_train.shape[0]*100
        ))

        print('total picked num kmeans={}, unripe={}, ripe={}, overripe={}, rate={:.2f}%'.format(
            np.sum(picked_num_kmeans), picked_num_kmeans[0], picked_num_kmeans[1], picked_num_kmeans[2], np.sum(picked_num_kmeans) / y_train.shape[0] * 100
        ))




