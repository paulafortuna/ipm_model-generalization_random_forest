# ipm_model-generalization_random_forest

In this repository we have the scripts that support the third experiment of our paper:

The complete experiment pipeline for this paper can be found [here](https://github.com/paulafortuna/IP-M_abusive_models_generalize).

In this experiment we aim at systematically study which model and dataset features lead to a better generalization in abusive language-related models, we run an experiment on the relation between the performance when applying BERT, ALBERT, fastText, and SVM and 16 prominent features of the models and datasets considered in the literature as good generalization predictors (e.g number of instances, dataset class). 

We group the 1698 binary BERT/ALBERT/fastText/SVM  models into models that generalize better (those with an F1 score >= 0.70; 136 in total) and models that generalize worse (those with an F1 score < 0.70; 1562 in total) The goal is to train a classifier on the above 16 features to predict whether a model belongs to the better generalizing models or worse generalizing models. As classifier, we use a Random Forest with 50 estimators with 5 Fold cross-validation. We have chosen Random Forest since it is a general-purpose classifier with weak statistical assumptions. To rank the different features used for classification, we use the [permutation feature importance algorithm](https://explained.ai/rf-importance/index.html). 

The code is organized as follows:

2020-10-21_dataset_ready_to_analyse.csv - The dataset that we achieved by following the code in the previous step of our pipeline. It contains the data about every SVM, fastText, BERT and ALBERT models we have from our previous experiments. 

random_forest_analysis.py  - It is a script for running a Random Forest model on our dataset


03_compute_model_performances.py - It is a script for computing Precision, Recall and F1 scores for every model.

04_compute_F1_table.py - It is a script for summarizing the F1 scores in one csv that will then be used in our paper results.
