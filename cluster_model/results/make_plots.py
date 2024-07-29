import os
import json

import matplotlib
import matplotlib.pyplot as plt

EVAL_DATA_FILE = "loss_eval_data.json"
CLUSTERING_RESULTS_FILE = "clustering_results.json"


def make_plots(filename, models, no_of_batches, show_legend=True):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.5)

    for idx, model_result_folder in enumerate(models):
        ax = axs[idx // 2, idx % 2]
        model_name = model_result_folder[3:]

        with open(f"{model_result_folder}/{CLUSTERING_RESULTS_FILE}") as f:
            clustering_results = json.load(f)
        print_model_result_data(model_name, clustering_results)

        with open(f"{model_result_folder}/{EVAL_DATA_FILE}") as f:
            loss_eval_data = json.load(f)

        title = (model_name
                 .replace("-", " ")
                 .replace("loss", "$\\ell$")
                 .replace("_star", "*")
                 .capitalize())
        ax.set_title(title, fontsize=12, pad=14)

        for level, loss_array in enumerate(loss_eval_data):
            accuracy = []
            batches = []
            for i, datapoint in enumerate(loss_array):
                accuracy.append(datapoint[1])
                batches.append(no_of_batches * i + 1)

            ax.plot(batches, accuracy, label=f"level {level}")

        ax.set_xlabel("Batches")
        ax.set_ylabel("Accuracy")
        ax.set_ylim([0, 1])
        ax.grid()
        if show_legend:
            ax.legend()
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(format_number))

    for idx in range(len(models), 4):
        axs[idx // 2, idx % 2].set_visible(False)

    fig.show()
    fig.savefig(filename)


def format_number(x, _):
    if x == 0:
        return "0"
    return f"{int(x // 1000)}k"


def print_model_result_data(model_name, clustering_results):
    print("*" * 100)
    print(model_name)
    print("*" * 100)
    print(f"Accuracy: {clustering_results['Full_Hierarchy']['Accuracy']}")

    for result in clustering_results:
        if result[:5] == "Level":
            print(f"Accuracy: {result} {clustering_results[result]['Accuracy_Kept']}")

    print()

    for result in clustering_results:
        print(result)
        print("-" * 100)

        if result[:5] == "Level":
            for cluster in clustering_results[result]["Clusters"]:
                examples = clustering_results[result]["Clusters"][cluster]["Sentences"][:20]
                input_text = ""

                for text in examples:
                    input_text += text
                    input_text += "\n"

                print(cluster, clustering_results[result]["Clusters"][cluster]["Total"])
                print("pos", clustering_results[result]["Clusters"][cluster]["Total_1"])
                print("neg", clustering_results[result]["Clusters"][cluster]["Total_0"])
                print("biased", clustering_results[result]["Clusters"][cluster]["Biased"])
                print(input_text)


folders = os.listdir()
folders.sort()

hierarchical_models = []
non_hierarchical_models = []

for folder in folders:
    if folder.startswith("e5-hierarchical"):
        hierarchical_models.append(folder)
    elif folder.startswith("e5"):
        non_hierarchical_models.append(folder)

make_plots(f"hierarchical_model_accuracies.png", hierarchical_models, show_legend=True, no_of_batches=1200)
make_plots(f"non_hierarchical_model_accuracies.png", non_hierarchical_models, show_legend=False, no_of_batches=1200)
