import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import optuna

# Load your data here
with open('embedding_labels.pkl', 'rb') as f:
    embedding_labels = pickle.load(f)

embeddings = np.array([item[0] for item in embedding_labels])
labels = [item[1] for item in embedding_labels]

# Create a dictionary to map class names to indices
unique_labels = list(set(labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

# Convert labels to numerical format
y = np.array([label_to_index[label] for label in labels])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': len(unique_labels),
        'metric': 'multi_logloss',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 10, 200),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    num_round = trial.suggest_int('num_round', 50, 300)
    bst = lgb.train(params, train_data, num_round)

    y_pred = bst.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test, y_pred_classes)
    return accuracy

# Run the optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print('Best hyperparameters: ', study.best_params)
print('Best accuracy: ', study.best_value)

# Train the final model with the best hyperparameters
best_params = study.best_params
best_params.update({
    'objective': 'multiclass',
    'num_class': len(unique_labels),
    'metric': 'multi_logloss',
    'verbose': -1
})

train_data = lgb.Dataset(X_train, label=y_train)
num_round = best_params.pop('num_round')
bst = lgb.train(best_params, train_data, num_round)

# Make predictions and calculate accuracy
y_pred = bst.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_pred_labels = [index_to_label[idx] for idx in y_pred_classes]
accuracy = accuracy_score([index_to_label[idx] for idx in y_test], y_pred_labels)
print(f'Accuracy: {accuracy}')

# Save the model and label mappings
with open('document_classification_model.pkl', 'wb') as f:
    pickle.dump(bst, f)

with open('label_mappings.pkl', 'wb') as f:
    pickle.dump((label_to_index, index_to_label), f)

print("Model and label mappings saved.")