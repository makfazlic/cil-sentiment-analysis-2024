# CIL Tweet Sentiment Analysis
## The Data
The Twitter data used can be downloaded at [Kaggle](https://www.kaggle.com/competitions/ethz-cil-text-classification-2024/data). Download and put the files `train_neg_full.txt` and `train_pos_full.txt` into the folder `data`.
## Baseline Experiments
The baseline experiments are included in the notebook `baselines.ipynb` to train some basic models on the data. The baseline mentioned in the report uses the BoW embedding with 5000 features and Logistic Regression as the model.
## Data Exploration
The notebook `data_exploration.ipynb` includes various models and preprocessing procedures. This was used to determine which preprocessing steps to take in the final model. Some results of these preliminary experiments are shown in the graph `data_exploration.png`.
## Clustering Model
### Training
To train the clustering model, run `bias_is_all_you_need.py`. This saves in the folder `.\local_models\own_models` the saving points of the different models as well as the configuration to run the hierarchical steps. This further generates the file `loss_eval_data.json` which includes a 3D array where the first dimension is the hierarchical level the second dimension represents the progression of the training and the 3 dimensions save in the first element the loss and in the second element the accuracy.
### Evaluation
After training, run `analyze_results.py` to evaluate the trained models extensively. This script uses the files generated in `.\local_models\own_models`. It generates the file `Determine_Clustering_Results.json` which includes the final accuracy evaluated on 100000 sentences, accuracy on the different levels, and information on the different clusters found. Note: `Determine_Clustering_Results.json` includes all sentences evaluated. Therefore it is not that readable. 
The results from our experiments are saved in `work_files/results`. This folder includes the following subfolders:
- `e5-baseline`: fine-tuned intfloat/multilingual-e5-large-instruc model
- `e5-clustering-with-loss`: no hierarchical algorithm + simple bias loss
- `e5-clustering-with-loss_star`: no hierarchical algorithm + use the additional distance loss
- `e5-hierarchical-clustering-with-loss`: hierarchical algorithm + simple bias loss
- `e5-hierarchical-clustering-with-loss_star`: hierarchical algorithm  + use the additional distance loss
- `e5-hierarchical-clustering-with-loss-and-reloading`: hierarchical algorithm + reload pre-trained model each level + simple bias loss
- `e5-hierarchical-clustering-with-loss_star-and-reloading`: hierarchical algorithm + reload pre-trained model each level + use the additional distance loss

The file `make_plots.py` creates plots and prints results based on this experimental data. To run it, first navigate to `work_files/results` and unpack the `results.rar` file. Select the option "extract here", so that the files are placed into the correct directories.
### Creation of the Submission
After training the model, run `make_submission.py` to generate the submission with the given test data. This script uses the files generated in `.\local_models\own_models` and generates the file `my_submission.csv` which is our submission to the Kaggle competition.
