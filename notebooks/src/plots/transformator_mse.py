import matplotlib.pyplot as plt
import numpy as np

def plot_mse(mses_dataset, stds_dataset, x_labels):
    n_bars = len(mses_dataset)
    xval = np.arange(n_bars)

    colors = ["r", "g", "b", "orange", "black"]

    # plot diabetes results
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(121)
    for j in xval:
        ax1.barh(
            j,
            mses_dataset[j],
            xerr=stds_dataset[j],
            color=colors[j],
            alpha=0.6,
            align="center",
        )

    ax1.set_title("Transformation Techniques")
    ax1.set_xlim(left=np.min(mses_dataset) * 0.9, right=np.max(mses_dataset) * 1.1)
    ax1.set_yticks(xval)
    ax1.set_xlabel("MSE")
    ax1.invert_yaxis()
    ax1.set_yticklabels(x_labels)