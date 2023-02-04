# NQE: N-ary Query Embedding for Complex Query Answering over Hyper-relational Knowledge Graphs

This folder contains the code used for the experiments in the paper

## Requirements

We developed our model using Python 3.7.10. Other version may also work.

First, please ensure that you have properly installed pytorch in your environment.

### Configure the Dataset

We tested the effectiveness of our model on two datasets, including the WD50K-NFOL dataset and the WD50K-QE dataset.

WD50K-NFOL is a hyper-relational dataset we created covering logical operations including conjunction, disjunction and negation as well as combined operations. 
WD50K-QE dataset is a dataset created with the multi-hop reasoning method starqe. It covering multi-hop reasoning with conjunction logic operation. To distinguish it from our proposed WD50K-NFOL dataset, we call it "starqe" in the code.

As a first step, you need to extract the dataset you want to use from the zip file, and place it in the "data" folder. Make sure you have the "tasks" folder and the "vocab.txt" in the folder. The "tasks" folder contains the json files of the preprocessed queries data. The "vocab.txt" contains all entities and relations that appear in the dataset. 

using WD50K-NFOL dataset
```bash
unzip WD50K-NFOL_dataset.zip
cp -r wd50k_nfol nge/data/
```

or

using starqe dataset
```bash
unzip WD50K-QE_dataset.zip
cp -r starqe nge/data/
```

After that, select the dataset you want to train or test by changing the argument "dataset".

## Running Training and Prediction ##

You can train query embedding model using "src/map_iter_qe.py" by setting argument "do_learn" True.

With default settings, our model using all types of queries to train the model.

The number of queries used for training does not exceed 100000 per type.

You can also choose to train our model with only a certain type of queries by changing the "train_tasks" option

## Running only prediction ##

You can only run prediction using "src/map_iter_qe.py" by with argument "do_learn" False and argument "do_predict" True.

In this case, you need to select the ckpts file you want to use and configure the "prediction_ckpt" argument as you want.

### Argument Description

We provide a more detailed and complete command description for training and testing the model:

| Parameter name | Description of parameter |
| --- | --- |
| dataset           | The dataset name |
| max_dataset_len   | Max queries number for training |
| train_tasks       | Types of queries to train model |
| validation_tasks  | Types of queries to evaluate model on validation set |
| test_tasks        | Types of queries to evaluate model on test set |
| valid_eval_tasks  | Tasks of queries to evaluate model on validation set, and calculate scores to finding best model hyperparameters |
| prediction_tasks  | Types of queries to run prediction |
| train_shuffle     | Whether choose to shuffle data in training process |
| checkpoint_dir    | Location of model checkpoints |
| prediction_ckpt   | ckpt filename. ckpt file contains saved model parameters |
| relative_validation_gt_dir | relative path for data folder containing groundtruth for validation set. Our model use groundtruth loaded from this folder to filter ranks and get final evaluation results on validation set. |
| relative_test_gt_dir | relative path for data folder containing groundtruth for test set. Our model use groundtruth loaded from this folder to filter ranks and get final evaluation results on test set. |
| train_shuffle     | Whether choose to shuffle data in training process |
| use_cuda          | Whether to use gpu |
| gpu_index         | The gpu no, used for training and prediction  |
| do_learn          | Whether to train model (defaults to `True`) |
| do_predict        | Whether to use model to predict (defaults to `True`) |
| epochs | Train epochs |
| batch_size | The batch size of model input queries |
| learning_rate | Optimizer learning rate |
| num_hidden_layers | Number of hidden layers in sequence encoder in projection operation |
| num_attention_heads   | Number of attention heads used in sequence encoder in projection operation |
| hidden_size       | Hidden size of fuzzy representation |
| intermediate_size | Intermediate size of fully connected layer in sequence encoder |
| hidden_dropout_prob   | Hidden dropout ratio |
attention_dropout_prob  | Attention dropout ratio |
| num_edges         | Number of edge types. Used for generating edge bias |
| entity_soft_label | Label smoothing rate for entities |
| L_config          | Encoder config. Configuration controls whether to use node bias and edge bias in sequence encoder |
| start_evaluation_epochs   | Epoch index to start evaluation |
| interval_evaluation_epochs | Epoch interval for evaluation |

