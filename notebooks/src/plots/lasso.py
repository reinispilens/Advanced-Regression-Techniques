import matplotlib.pyplot as plt

def coordinate_descent(lasso):
    ymin, ymax = 2300, 3800
    plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
    plt.plot(
        lasso.alphas_,
        lasso.mse_path_.mean(axis=-1),
        color="black",
        label="Average across the folds",
        linewidth=2,
    )
    plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")

    plt.ylim(ymin, ymax)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Mean square error")
    plt.legend()
    _ = plt.title(
        f"Mean square error on each fold: coordinate descent (train time: {fit_time:.2f}s)"
    )