import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load IMDb dataset (preprocessed)
num_words = 10000
maxlen = 200
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)

# Pad sequences
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Build model
model = keras.Sequential([
    layers.Embedding(num_words, 32, input_length=maxlen),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.2)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# Save model
model.save("model/sentiment_model.h5")
print("✅ Model saved successfully!")
model.save("model/sentiment_model.keras")

