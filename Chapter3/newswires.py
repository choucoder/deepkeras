import numpy as np
from keras.datasets import reuters
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def one_hot_encoding(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


# Facing: frente a
# Classifying newswires (Cablres de noticias)
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

maximun = max([max([word for word in sample]) for sample in train_data])
word_index = reuters.get_word_index()
print("word_index: {}".format(len(word_index)))
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswires = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_newswires)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = one_hot_encoding(train_labels)
one_hot_test_labels = one_hot_encoding(test_labels)

#one_hot_train_labels = to_categorical(one_hot_train_labels)
#one_hot_test_labels = to_categorical(one_hot_test_labels)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10000,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[: 1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[: 1000]
partial_y_train = one_hot_train_labels[1000: ]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)
print("[Loss, accuracy]: {}".format(results))

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()