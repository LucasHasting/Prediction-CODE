# Prediction-CODE
For this project, we use machine learning models to predict pathogenicity associated with SERPINA1 (which is associated with AAT). The dataset comes from ensembl (P01009). We use "SIFT", "PolyPhen", "REVEL", "MetaLR", and "Mutation Assessor" to predict "Clin. Sig." (see clean_data.py).

Note: Some models will be difficult to replicate due to the randomness involved. The models used are saved in the models folder.

## Table of Contents

1. [Files](#files)  
2. [Build Instructions](#build-instructions)  
3. [Rational for Models](#rational-for-models)  
4. [Suggested Future Work](#suggested-future-work)  
5. [Sources](#sources)  

## Files

| File                         | Description                                                          |
|------------------------------|----------------------------------------------------------------------|
| models/*.pkl                        | Model binary files.                                           |
| data/*.pkl                          | Dataframe binary files.                                       |
| data/ensembl-export-serpina1.xlsx   | The full dataset.                                             |
| data/predictions_VUS.csv            | The dataset containing predictions for VUS using each model.  |
| clean_data.py                       | Cleans the full dataset and generates data *.pkl files.       |
| PNN.py                              | Trains the NN model.                                          |
| check_nn_params.py                  | Displays the parameters of the NN model model.                |
| param_search.py                     | Searches for the best parameters for the other models.        |
| models.py                           | Trains the other models.                                      |
| models_test.py                      | Tests all models and shows a CM for each model.               |
| classify.py                         | Generates predictions_VUS.csv.                                |
| poster.pdf                          | The poster used to present the project.                       |
| poster.zip                          | The source for poster.pdf.                                    |

## Build Instructions

First, ensure python is installed, it can be installed [here](https://www.python.org/downloads/) and, optionally, that git is installed, it can be installed [here](https://git-scm.com/install/). When you install python, make sure to add the installation to the PAtH environment variable. In the event that you did not do that, follow the instructions [here](https://phoenixnap.com/kb/linux-add-to-path) for Linux and [here](https://www.youtube.com/watch?v=9umV9jD6n80&feature=youtu.be) for Windows.

Next, use pip to install the following packages:

```sh
pip install scikit-learn
pip install torch
pip install pandas
pip install matplotlib
```

Now you can use python from the command line or open/execute each program using the IDE idle.

## Rational for Models
Since we are using binary classification, a sigmoid activation function is used on the output layer.

1 Hidden layer is used, there are few inputs, so additional layers are not needed.

The loss function used is log loss, this is done since we are doing binary classification.

The neural network is trained using backpropagation and optomized using stocastic gradient descent.

Anything else with the NN was chosen through experimentation.

Gini is used for the decision tree/random forest since we are using binary classification (easier to interpret).

Euclidean Distance is used for K-NN since all variables are normalized.

## Suggested Future Work
(1) Results from the decision tree/random forest can be used to make inferences on the various perdictors.

(2) Additional predictors can be explored.

(3) Additional layers/activation functions can be experimented within the MLP, and other types of neural networks could be explored.

(4) The models can be tested on datasets associated with other genes/proteins.

(5) New models can be trained on datasets associated with other genes/proteins.

(6) Possibly combine (4) and (5) to create models that generalize to predicting pathogenicity, regardless of the gene/protein.

## Sources
(1) https://builtin.com/machine-learning/common-loss-functions

(2) https://www.deeplearningbook.org/ (chapters 5/6 are the most helpful)

(3) https://mlbenchmarks.org/04-holdout-method.html 

(4) https://en.wikipedia.org/wiki/Loss_functions_for_classification 

(5) https://scikit-learn.org/stable/ 

(6) https://docs.pytorch.org/docs/stable/index.html 

(7) https://en.wikipedia.org/wiki/Decision_tree_learning 

(8) https://en.wikipedia.org/wiki/Stochastic_gradient_descent 
