# Binary Classification with a Neural Network (Keras)

A beginner practice project where I learn deep learning by building a simple neural network that predicts a binary target (0 or 1) from 11 input features.

> **Note:** This is a learning project, not a production model.

## What I Practiced

- Splitting data into train / validation / test sets
- Scaling features with `StandardScaler`
- Building a model with `Sequential` and `Dense` layers
- Compiling with `binary_crossentropy`, the `adam` optimizer, and the `accuracy` metric
- Training with `model.fit()` and reading loss and accuracy
- Spotting overfitting and underfitting with loss curves
- Using `EarlyStopping`
- Finding the best threshold with the ROC curve and AUC
- Evaluating with a confusion matrix and classification report

## Libraries Used

- `pandas`: loading and handling the data
- `numpy`: numerical operations
- `matplotlib`: plotting loss, accuracy, and ROC curves
- `scikit-learn`: train/test split, scaling, metrics
- `tensorflow` / `keras`: building and training the neural network

## Dataset

- 12 columns in total: 11 input features and 1 target
- Target: binary (0 or 1)
- Source: *https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction/code*

## Model Architecture

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=11))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## Project Workflow

1. Load and explore the data
2. Separate features (`X`) and target (`y`)
3. Split into train, validation, and test sets
4. Scale the features (fit the scaler on train data only)
5. Build, compile, and train the model
6. Plot loss and accuracy to check for overfitting
7. Find the best threshold with the ROC curve
8. Evaluate on the test set

## Results

Evaluated on the test set (2,000 rows).

| Metric | Value |
|---|---|
| Train accuracy | *0.8700* |
| Validation accuracy | *0.8569* |
| Test accuracy | 0.8425 (84.25%) |
| AUC | *0.85* |
| Best threshold | *0.27114546298980713* |

### Confusion Matrix

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 1419 | 188 |
| **Actual 1** | 127 | 266 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 | 0.92 | 0.88 | 0.90 | 1607 |
| 1 | 0.59 | 0.68 | 0.63 | 393 |
| **Accuracy** | | | 0.84 | 2000 |
| Macro avg | 0.75 | 0.78 | 0.76 | 2000 |
| Weighted avg | 0.85 | 0.84 | 0.85 | 2000 |

### Observations

- The data is imbalanced: about 80% of the test rows are class 0 and about 20% are class 1.
- The model is much better at class 0 (F1 0.90) than at class 1 (F1 0.63).
- Of the 393 actual class 1 rows, the model found 266 (recall 0.68) and missed 127.



## What I Learned

- Always compare accuracy with a baseline, especially if the classes are imbalanced
- Never use test data to choose settings such as the threshold
- Watching validation loss is the easiest way to catch overfitting
- Most of my errors were small typos, and reading the last line of the error helps a lot

## Next Steps

- Try more layers and neurons
- Add `Dropout` to reduce overfitting
- Try other datasets and multi-class problems

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.