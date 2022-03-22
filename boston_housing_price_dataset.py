# The Boston Housing Price dataset
from keras.datasets import boston_housing
from keras import layers, models, optimizers, losses
import numpy as np
import matplotlib.pyplot as plt

# Loading the Boston housing dataset
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print("train data shape", train_data.shape)
print("test data shape", test_data.shape)
print("train targets: ", train_targets)

# Preparing the data
mean = np.mean(train_data, axis=0)
print("mean: ", mean)
train_data -= mean
std = np.std(train_data, axis=0)
train_data /= std

test_data -= mean
test_data /= std

# Building your network

model = models.Sequential()
model.add(layers.Dense(64, activation="relu",
          input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(1))

model.compile(optimizer=optimizers.adam_v2.Adam(learning_rate=0.001),
              loss=losses.categorical_crossentropy, metrics=["acc"])

# Validating your approach using K-fold validation
# K-fold validation

k = 4
num_val_samples = int(len(train_data)/k)
num_epochs = 100
all_scores = []

for i in range(k):
    print("processing fold #", i)
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i*num_val_samples], train_data[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i*num_val_samples], train_targets[(i+1)*num_val_samples:]], axis=0)

    model.fit(partial_train_data, partial_train_targets, batch_size=1,
              epochs=num_epochs)  # (in silent mode,verbose = 0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print("ortalama score: ", np.mean(all_scores))

# Saving the validation logs at each fold
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print("processing fold #", i)
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i*num_val_samples], train_data[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i*num_val_samples], train_targets[(i+1)*num_val_samples:]], axis=0)
    history = history = model.fit(partial_train_data, partial_train_targets,
                                  validation_data=(val_data, val_targets),
                                  epochs=num_epochs, batch_size=1,)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# Plotting validation scores
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# Plotting validation scores, excluding the first 10 data points


def smooth_curve(points, factor=0.9):
    assert len(points) > 1
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# Training the final model
model.fit(train_data, train_targets,
          epochs=80, batch_size=16)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
