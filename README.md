# Prediction-CODE

Note: Some models will be difficult to replicate due to the randomness involved. The models used are saved in the models folder.

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

## Rational for NN, Decision Tree/Random Forest, K-NN
Since we are using binary classification, a sigmoid activation function is used on the output layer.

1 Hidden layer is used, there are few inputs, so additional layers are not needed.

The loss function used is log loss, this is done since we are doing binary classification.

The neural network is trained using backpropagation and optomized using stocastic gradient descent.

Anything else with the NN was chosen through experimentation.

Gini is used for the decision tree/random forest since we are using binary classification (easier to interpret).

Euclidean Distance is used for K-NN since all variables are normalized.

## Sources
(1) https://builtin.com/machine-learning/common-loss-functions

(2) https://www.deeplearningbook.org/contents/mlp.html

(3) https://mlbenchmarks.org/04-holdout-method.html 

(4) https://en.wikipedia.org/wiki/Loss_functions_for_classification 

(5) https://scikit-learn.org/stable/ 

(6) https://docs.pytorch.org/docs/stable/index.html 

(7) https://en.wikipedia.org/wiki/Decision_tree_learning 

(8) https://en.wikipedia.org/wiki/Stochastic_gradient_descent 
