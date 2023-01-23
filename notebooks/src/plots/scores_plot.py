import matplotlib.pyplot as plt


def scores_plot(names, results):
        # set the figure size and layout
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10,16))
    fig.tight_layout()

    # iterate through each of the regressor names
    for i, name in enumerate(names):
        # select the appropriate subplot
        ax = axes[i//2, i%2]
        # set the title for the subplot
        ax.set_title(name)
        # plot the histograms for the test and train scores
        ax.hist(results[["test_neg_mean_absolute_error", "train_neg_mean_absolute_error"]][results['model'] == name], bins=20, label=['test', 'train'])
        ax.legend()
    plt.show()