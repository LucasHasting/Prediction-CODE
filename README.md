# Prediction-CODE

Note: Random Forest Classifier is involved with randomness and will most likely give a different model when replicating results.

| File                         | Description                                                  |
|------------------------------|--------------------------------------------------------------|
| *.pkl                        | Model/data binary files.                                     |
| ensembl-export-serpina1.xlsx | The full dataset.                                            |
| predictions_VUS.csv          | The dataset containing predictions for VUS using each model. |
| readFile.py                  | Cleans the full dataset and generates data *.pkl files.      |
| PNN.py                       | Trains the NN model                                          |
| param_search.py              | Searches for the best parameters for the other models        |
| models.py                    | Trains the other models                                      |
| models_test.py               | Tests all models and shows a CM for each model               |
| classify.py                  | Generates predictions_VUS.csv                                |

## Rational for NN
Since we are using binary classification, a sigmoid activation function is used on the output layer.

1 Hidden layer is used, there are few inputs, so additional layers are not needed.

The loss function used is log loss, this is done since we are doing binary classification.

The neural network is trained using backpropagation and optomized using stocastic gradient descent.

## Sources for Rational
(1) https://builtin.com/machine-learning/common-loss-functions
(2) https://www.deeplearningbook.org/contents/mlp.html
