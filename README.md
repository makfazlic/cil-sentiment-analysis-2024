# CIL Tweet Sentiment Analysis
## The Data
The twitter data used can be downloaded at [[link]].
## Baseline Experiments
The baseline experiments include a Jupyter notebook to train some basic models on the data. The baseline mentioned in the report uses the BoW embedding with 5000 features and Logistic Regression as the model. To be able to run the code, the folder `twitter-datasets` needs to be put into the folder `baselines`.
## Preliminary Experiments
## Clustering Model
The folder `cluster_model` contains our final model and experiments.
### Training
The folder `training` includes the code for training and evaluating the hierarchical algorithm for reproduction reasons. It further includes the code for creating the submission.
To run the code follow the following steps in the folder `training`:

1. Download and put the files `train_neg_full.txt` and `train_pos_full.txt` into the folder `training`.

2. Run `Bias_is_all_you_need.py`.
   This saves in the folder `.\local_models\own_models` the saving points of the different models as well as the configuration to run the hierarchical steps.
   It further generates in the file `loss_eval_data.json` which includes a 3D Array where the first dimension is the hierarchical level the second dimension represents the progression of the training and the 3 dimensions saves in the first element the loss and in the    
second element the accuracy.

4. After completion of the first step run `Analyze_Results.py` to evaluate the trained models extensively
   This script uses the files generated in `.\local_models\own_models`.
   It  generates the file `Determine_Clustering_Results.json` which includes the final accuracy evaluated on 100000 sentences, accuracy on the different levels, and information on the different clusters found.
   Note: `Determine_Clustering_Results.json` includes all sentences evaluated. Therefore it is not that readable. We provided a Python file that presents the results more nicely. This can be found in the folder `results`.

5. After completion of the first step run `Make_Submission.py` to generate the submission.
   This script uses the files generated in `.\local_models\own_models`
   This script generates the file `my_submission.csv` which is our submission to the Kaggle competition.

### Results
The folder `results` includes the data of experiments reported in the paper.
These files can be generated by adapting the genome in `./training/Bias_is_all_you_need.py` and follow the steps above.
This folder includes the following subfolders:
- e5-baseline: fine-tuned intfloat/multilingual-e5-large-instruc model
- e5-clustering-with-loss: no hierarchical algorithm + simple bias loss
- e5-clustering-with-loss_star: no hierarchical algorithm + use the additional distance loss
- e5-hierarchical-clustering-with-loss: hierarchical algorithm + simple bias loss
- e5-hierarchical-clustering-with-loss_star: hierarchical algorithm  + use the additional distance loss
- e5-hierarchical-clustering-with-loss-and-reloading: hierarchical algorithm + reload pre-trained model each level + simple bias loss
- e5-hierarchical-clustering-with-loss_star-and-reloading: hierarchical algorithm + reload pre-trained model each level + use the additional distance loss
