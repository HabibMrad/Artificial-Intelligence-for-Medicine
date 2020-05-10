import matplotlib.pyplot as plt
import numpy as np

def display_images(num_rows, num_cols, X, Y=None):
    fig, axis = plt.subplots(nrows=num_rows, ncols=num_cols,
                             figsize=(num_cols*4, 8))
    counter = 0
    for i in range(num_rows):
        for j in range(num_cols):
            axis[i][j].imshow(X[counter, :, :, 0], cmap=plt.cm.bone)
            if type(Y) == np.ndarray:
                axis[i][j].imshow(Y[counter, :, :, 0], cmap='Reds', alpha=0.35)
            counter += 1
    plt.tight_layout()
    plt.show()
