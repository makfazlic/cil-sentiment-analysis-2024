import os
import json
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

EVAL_DATA_FILE = "loss_eval_data.json"
CLUSTERING_RESULTS_FILE = "clustering_results.json"


def make_plots(filename, models, show_legend=True):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.5)

    for idx, model_result_folder in enumerate(models):

        ax = axs[idx // 2, idx % 2]
        model_name = model_result_folder['folder'].replace('e5-', '')

        clustering_file = f"{results_folder}/{model_result_folder['folder']}/{CLUSTERING_RESULTS_FILE}"
        eval_data_file = f"{results_folder}/{model_result_folder['folder']}/{EVAL_DATA_FILE}"

        if Path(clustering_file).is_file():
            with open(clustering_file) as f:
                clustering_results = json.load(f)
            print_model_result_data(model_name, clustering_results)

        with open(eval_data_file) as f:
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
                batches.append((model_result_folder['xsize'] * i) // len(loss_array) + 1)

            ax.plot(batches, accuracy, label=f"level {level}")

        ax.set_xlabel(model_result_folder['xlabel'])
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

    filepath = f'./work_files/plots/{filename}'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath)


def format_number(x, _):
    if x < 1000:
        return x
    if x < 10000:
        return f"{x / 1000:.1f}k"
    if x < 1000000:
        return f"{int(x // 1000)}k"
    return f"{x / 1000000:.1f}M"


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


results_folder = './cluster_model/results/'
# results_folder = './work_files/results/'
folders = os.listdir(results_folder)
folders.sort()

hierarchical_models = []
non_hierarchical_models = []

for folder in folders:
    if folder.startswith("e5-hierarchical"):
        hierarchical_models.append({'folder': folder, 'xsize': 12000, 'xlabel': 'Batches'})
    elif folder.startswith("e5"):
        non_hierarchical_models.append({'folder': folder, 'xsize': 36000, 'xlabel': 'Batches'})
    elif folder.startswith("Logistic"):
        non_hierarchical_models.append({'folder': folder, 'xsize': 200000, 'xlabel': 'Dataset size'})

make_plots(f"work_files/results/hierarchical_model_accuracies.png", hierarchical_models, show_legend=True)
make_plots(f"work_files/results/non_hierarchical_model_accuracies.png", non_hierarchical_models, show_legend=False)
