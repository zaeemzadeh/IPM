from ipm import ipm
import numpy as np

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load data
    X, y = fetch_openml('mnist_784', version=1, cache=True, return_X_y=True)
    X = X / 255.

    # select a random subset of data (for faster execution)
    # idx = np.random.permutation(range(len(y)))[:10000]
    # X = X[idx]
    # y = y[idx]

    # select samples using ipm
    selected_idx = ipm(list(X), n=10)

    # plot selected samples
    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        image = X[selected_idx[i]].reshape(28, 28)
        plt.imshow(image, cmap=plt.cm.gray_r)
        plt.title('Label: ' + y[selected_idx[i]])

    plt.show()