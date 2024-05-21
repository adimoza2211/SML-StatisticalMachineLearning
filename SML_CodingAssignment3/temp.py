import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

# Load MNIST dataset
data = np.load('mnist.npz')
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

# Preprocess data
x_train = (x_train.reshape(60000, 784)) / 255
x_test = (x_test.reshape(10000, 784)) / 255

# Compute PCA transformation
def getPCA_Data(data, mean):
    data = data - mean
    cov = data.T @ data / 60000
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = np.argsort(eigenvalues)[::-1]
    U = eigenvectors[:, idx]
    U = np.real(U)
    U = U[:, :5]
    Y = U.T @ data.T
    Y = Y.T
    Y = np.real(Y)
    return Y, U

# Gradient boosting for regression using absolute loss
def gradient_boosting(train_data, train_labels, val_data, val_labels, num_trees=100):
    train_mean_fulldim = np.mean(train_data, axis=0)
    train, U = getPCA_Data(train_data, train_mean_fulldim)
    val = val_data - train_mean_fulldim
    val = val @ U
    
    # Initialize variables
    models = []
    preds_arr = []
    mses = []
    learning_rate = 0.01

    for i in range(num_trees):
        # Find best split for current iteration
        best_split, best_split_dim, _, best_left_mean, best_right_mean = find_best_split(train, train_labels)
        models.append((best_split, best_split_dim, best_left_mean, best_right_mean))

        # Compute residuals
        residuals = compute_residuals(train, train_labels, best_split, best_split_dim, best_left_mean, best_right_mean)

        # Predict on validation set
        preds = predict_labels(val, best_split, best_split_dim, best_left_mean, best_right_mean)
        preds_arr.append(preds)

        # Compute MSE on validation set
        mse = compute_MSE(preds_arr, val_labels)
        mses.append(mse)

        # Update labels using residuals
        train_labels -= learning_rate * residuals

        print(f"Iteration {i+1}, Validation MSE: {mse}")

    # Plot MSE over iterations
    plt.plot(mses)
    plt.xlabel('Number of trees')
    plt.ylabel('MSE')
    plt.show()

    # Return models and final validation MSE
    return models, mses[-1]

# Find best split based on absolute loss
def find_best_split(data, labels):
    min_ssr = float('inf')
    best_split = 0
    best_split_dim = 0
    best_left_mean = 0
    best_right_mean = 0

    for i in range(5):
        unique_vals = np.unique(data[:, i])
        splits = (unique_vals[:-1] + unique_vals[1:]) / 2

        for split in splits:
            left_indices = data[:, i] < split
            right_indices = data[:, i] >= split

            left_labels = labels[left_indices]
            right_labels = labels[right_indices]

            left_mean = np.mean(left_labels)
            right_mean = np.mean(right_labels)

            left_ssr = np.sum(np.abs(left_labels - left_mean))
            right_ssr = np.sum(np.abs(right_labels - right_mean))
            ssr = left_ssr + right_ssr

            if ssr < min_ssr:
                min_ssr = ssr
                best_split = split
                best_split_dim = i
                best_left_mean = left_mean
                best_right_mean = right_mean

    return best_split, best_split_dim, min_ssr, best_left_mean, best_right_mean

# Compute residuals using absolute loss
def compute_residuals(data, labels, split, split_dim, left_mean, right_mean):
    residuals = np.zeros(len(labels))
    for i in range(len(labels)):
        if data[i][split_dim] < split:
            residuals[i] = labels[i] - left_mean
        else:
            residuals[i] = labels[i] - right_mean
    return residuals

# Predict labels for given dataset
def predict_labels(data, split, split_dim, left_mean, right_mean):
    preds = np.where(data[:, split_dim] < split, left_mean, right_mean)
    return preds

# Compute MSE between predicted and actual labels
def compute_MSE(preds_arr, labels):
    mse = np.mean((labels - np.sum(preds_arr, axis=0)) ** 2)
    return mse

# Prepare training and validation data
class0 = x_train[y_train == 0]
class1 = x_train[y_train == 1]
labels = np.concatenate((np.full(class0.shape[0], -1), np.full(class1.shape[0], 1)))

val_class0 = class0[:1000]
val_class1 = class1[:1000]
val_labels = np.concatenate((np.full(1000, -1), np.full(1000, 1)))

train_class0 = class0[1000:]
train_class1 = class1[1000:]

# Train gradient boosting model
models, val_mse = gradient_boosting(np.concatenate((train_class0, train_class1), axis=0), labels,
                                    np.concatenate((val_class0, val_class1), axis=0), val_labels)

print("Validation MSE:", val_mse)

# Prepare testing data
test_labels = np.concatenate((np.full(980, -1), np.full(1135, 1)))
test_data = np.concatenate((x_test[y_test == 0], x_test[y_test == 1]), axis=0)

# Predict labels on testing data
test_preds_arr = []
for model in models:
    split, split_dim, left_mean, right_mean = model
    preds = predict_labels(test_data, split, split_dim, left_mean, right_mean)
    test_preds_arr.append(preds)

test_preds_arr = np.array(test_preds_arr)
test_mse = compute_MSE(test_preds_arr, test_labels)

print("Test MSE:", test_mse)
