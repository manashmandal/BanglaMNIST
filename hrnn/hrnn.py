
from keras.models import Model

# Training parameters.
batch_size = 35
num_classes = 10
epochs = 300

img_rows = 48
img_cols = 48

X_train = get_train_X()
x_train = np.reshape(X_train, (X_train.shape[0], img_rows, img_cols, 1))
y_train = get_train_Y()

x_train, y_train = shuffle(x_train, y_train)

row, col, pixel = x_train.shape[1:]

x = Input(shape=(row, col, pixel))

# checkpoint
from keras.callbacks import ModelCheckpoint

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Embedding dimensions.
row_hidden = 64
col_hidden = 64

# Encodes a row of pixels using TimeDistributed Wrapper.
encoded_rows = TimeDistributed(LSTM(row_hidden))(x)

# Encodes columns of encoded rows.
encoded_columns = LSTM(col_hidden)(encoded_rows)

# Final predictions and model.
prediction = Dense(num_classes, activation='softmax')(encoded_columns)
model = Model(x, prediction)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Training.
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks_list,
          validation_split=.33)