import imdb
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import losses
from keras import metrics

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
        
    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Get word_index and decode reviews
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)
decoded_review = ''.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
)

# Preparing data into binary matrix
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Validation set
x_val = x_train[: 10000]
partial_x_train = x_train[10000: ]

y_val = y_train[: 10000]
partial_y_train = y_train[10000: ]

# Define the model

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(10000,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer=optimizers.RMSprop(lr=0.001),
    loss=losses.binary_crossentropy,
    metrics=['accuracy']
)

history = model.fit(x_train,
        y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test)

print(results)

print(model.predict(x_test))