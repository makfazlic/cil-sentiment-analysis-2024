# cil-sentiment-analysis-2024
## Overview
### Baseline Experiments
### Preprocessing Experiments
### Clustering Model
The folder `cluster_model` contains our model.
#### Training
The folder `training` includes the code for training and evaluating the hierarchical algorithm for reproduction reasons. It further includes the code for creating the Submission.
To run the code follow the following steps in the folder `training`:

1. Download the data at [[link]] and put the files `train_neg_full.txt` and `train_pos_full.txt` into the folder `training`.

2. Run `Bias_is_all_you_need.py`.
   This saves in the folder `.\local_models\own_models` the saving points of the different models as well as the configuration to run the hierarchical steps.
   It further generates in the file `loss_eval_data.json` which includes a 3D Array where the first dimension is the hirarchical Level the second dimension represents the progession of the training and in the 3 dimension saves in the first element the loss and in the second element the accuracy.
   Note: This code uses CPU so rerunning is not recommended as it Needs around 3 days to complete. We further did not upload the trained models as GitHub does not allow for such big files (over 1 GB)

3. After completion of the first step run `Analyze_Results.py` to evaluate the trained models extensively
   This script uses the files generated in `.\local_models\own_models`.
   It  generates the file `Determine_Clustering_Results.json` which includes the final accuracy evaluated on 100000 sentences, accuracy on the different levels and information on the different clusters found.
   Note: `Determine_Clustering_Results.json` includes all sentences evaluated. Therefore it is not that readable. We provided a python file which presents the results more nicely. This can be found in the folder `results`.

4. After completion of the first step run `Make_Submission.py` to generate the submission.
   This script uses the files generated in `.\local_models\own_models`
   This script generates the file `my_submission.csv` which is our submission to the Kaggle competition.

### Results
The folder `results` includes the data of experiments reported in the paper.
These files can be generated by adapting the genome in `./training/Bias_is_all_you_need.py` and follow the steps above.

This folder includes the following subfolders:
- cl0  : hierarchical algo  + use the additional distance loss (loss* instead of loss in the report) [results reported in the additional experiments]
- cl2  : hierarchical algo [The hierarchical algo results]
- cl3  : no hierarchical algo + use the additional distance loss (loss* instead of loss in the report) [results reported in the additional experiments]
- cl5  : no hierarchical algo + baseline (no clustering loss) [baseline]
- cl8  : hierarchical algo + reload pretrained model each level [results reported in the additional experiments]
- cl9  : no hierarchical algo + not baseline (using loss instead of loss* in report) [The bias loss in the report (without hierarchical)]
- cls0 : hierarchical algo + reload pretrained model each level + use the additional distance loss (loss* instead of loss in the report) [results reported in the additional experiments]

This folders include the following files:
Determine_Clustering_Results.zip : This should be first be unziped to run the python files described further down. This includes extended data on the final evaluation
loss_eval_data.json : This includes loss and accuracy data at different times in the training
… .png              : This includes the some graphs of the accuracy development through training and hierarchical steps.

It additionally includes the following files:
Make_Eval_Data.py         : This program returns a better structured output of accuracy and example sentences of differnt clusters of the different experiments. It additionally generates the graphs.
Make_Eval_Data_Output.txt : This is the output of the "Make_Eval_Data.py" file.
