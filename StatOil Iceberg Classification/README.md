# StatOil Iceberg Classification
This is a baseline model for the Kaggle contest which can be found [here]https://www.kaggle.com/c/statoil-iceberg-classifier-challenge
## Method
This baseline model is built by the ensemble of three Convolutional Neural Networks
* Model_1: CNN with inception module, trained on Augmented training data without angle info
* Model_2: CNN with inception module, trained on original data with angle info
* Model_3: CNN without inception module, trained on original data with angle info

Final result was an average of above three models, and had log-loss of 0.1968 on private leaderboard, which is a reasonable baseline model

Trained model weights and corresponding computational graphs are stored in three model_save folders respectively
