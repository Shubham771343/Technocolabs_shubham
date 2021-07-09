# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import regularizers

from tensorflow.keras import layers
from tensorflow.keras import losses

from collections import Counter


import pandas as pd
import numpy as np

import sklearn


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


import seaborn as sns

import pydot

electric_product = pd.read_csv("C:/Users/vikar/Desktop/Viraj Karkar/Technocolab/Supervised ML Task-1/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv", usecols=['reviews_text', 'reviews_rating','primaryCategories', 'id'])
electric_product1 = pd.read_csv("C:/Users/vikar/Desktop/Viraj Karkar/Technocolab/Supervised ML Task-1/1429_1.csv", usecols=['reviews_text', 'reviews_rating','primaryCategories', 'id'])
electric_product2 = pd.read_csv("C:/Users/vikar/Desktop/Viraj Karkar/Technocolab/Supervised ML Task-1/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv", usecols=['reviews_text', 'reviews_rating','primaryCategories', 'id'])


# %%
electric = pd.concat([electric_product,electric_product1,electric_product2])
electric


# %%

print(tf.__version__)
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)


# %%
if tf.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

else:
    print("Please install GPU version of TF")


# %%
def clean_text(text ): 
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' ' 
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    #print('cleaned:'+text1)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr if ( not w.isdigit() and  ( not w.isdigit() and len(w)>2))]) 
    
    return text2.lower()


# %%
print(electric.head(10))
print(len(electric))
print('Unique Products')
print(len(electric.groupby('id')))


# %%

electric.dropna(axis = 0, how ='any',inplace=True)
electric['reviews_text'] = electric['reviews_text'].apply(clean_text)
# electric['reviews_text'] = electric['reviews_text'].apply(remove_url)
electric['Num_words_text'] = electric['reviews_text'].apply(lambda x:len(str(x).split()))


# %%
print('-------Dataset --------')
print(electric['reviews_rating'].value_counts())
print(len(electric))
print('-------------------------')
max_electric_sentence_length  = electric['Num_words_text'].max()

print('Train Max Sentence Length :'+str(max_electric_sentence_length))


#all_sentences = train_data['text'].tolist() + test_data['text'].tolist()


# %%
electric['Num_words_text'].describe()


# %%
sns.set(style="whitegrid")
sns.boxplot(x=electric['Num_words_text'])


# %%
mask = (electric['Num_words_text'] < 100) & (electric['Num_words_text'] >=20)
electric_short_reviews = electric[mask]
print('No of Short reviews')
print(len(electric_short_reviews))

mask = electric['Num_words_text'] >= 100
electric_long_reviews = electric[mask]
print('No of Long reviews')
print(len(electric_long_reviews))


# %%
print(electric_short_reviews['Num_words_text'].max())


# %%
def get_sentiment(rating):
    if rating == 5 or rating == 4 or rating ==3:
        return 1
    else:
        return 0


# %%
electric_short_reviews['reviews_rating'].value_counts()
filtered_data = electric_short_reviews.groupby('id').filter(lambda x: len(x) >= 20)
print(len(filtered_data))
print(filtered_data ['reviews_rating'].value_counts())
filtered_data ['sentiment'] = filtered_data ['reviews_rating'].apply(get_sentiment)
#train_data = electric_short_reviews.sample(n=30000, random_state =0)
train_data = filtered_data[['reviews_text','sentiment']]
print('Train data')
print(train_data['sentiment'].value_counts())


#Create Test Data
mask = electric['Num_words_text'] < 100 
df_short_reviews = electric[mask]
filtered_data = electric_short_reviews.groupby('id').filter(lambda x: len(x) >= 10)
print(len(filtered_data))
print(filtered_data ['reviews_rating'].value_counts())
filtered_data ['sentiment'] = filtered_data ['reviews_rating'].apply(get_sentiment)
#train_data = df_short_reviews.sample(n=200000, random_state =0)
test_data = filtered_data[['reviews_text','sentiment']]
print('Test data')
print(test_data['sentiment'].value_counts())


# %%
train_data['sentiment'].value_counts()


# %%
X_train, X_valid, y_train, y_valid = train_test_split(train_data['reviews_text'].tolist(),                                                      train_data['sentiment'].tolist(),                                                      test_size=0.5,                                                      stratify = train_data['sentiment'].tolist(),                                                      random_state=0)


print('Train data len:'+str(len(X_train)))
print('Class distribution'+str(Counter(y_train)))
print('Valid data len:'+str(len(X_valid)))
print('Class distribution'+ str(Counter(y_valid)))


# %%
num_words = 90000

tokenizer = Tokenizer(num_words=num_words,oov_token="unk")
tokenizer.fit_on_texts(X_train)


print(str(tokenizer.texts_to_sequences(['xyz how are you'])))


# %%
x_train = np.array( tokenizer.texts_to_sequences(X_train) )
x_valid = np.array( tokenizer.texts_to_sequences(X_valid) )
x_test  = np.array( tokenizer.texts_to_sequences(test_data['reviews_text'].tolist()) )



x_train = pad_sequences(x_train, padding='post', maxlen=100)
x_valid = pad_sequences(x_valid, padding='post', maxlen=100)
x_test = pad_sequences(x_test, padding='post', maxlen=100)



train_labels = np.asarray(y_train )
valid_labels = np.asarray( y_valid)

test_labels = np.asarray(test_data['sentiment'].tolist())

print('Train data len:'+str(len(x_train)))
print('Class distribution'+str(Counter(train_labels)))

print('Validation data len:'+str(len(x_valid)))
print('Class distribution'+str(Counter(valid_labels)))

print('Test data len:'+str(len(x_test)))
print('Class distribution'+str(Counter(test_labels)))


train_ds = tf.data.Dataset.from_tensor_slices((x_train,train_labels))
valid_ds = tf.data.Dataset.from_tensor_slices((x_valid,valid_labels))
test_ds = tf.data.Dataset.from_tensor_slices((x_test,test_labels))


# %%
max_features =50000
embedding_dim =16
sequence_length = 100

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(max_features +1, embedding_dim, input_length=sequence_length,                                    embeddings_regularizer = regularizers.l2(0.005))) 
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.LSTM(embedding_dim,dropout=0.2, recurrent_dropout=0.2,return_sequences=True,                                                             kernel_regularizer=regularizers.l2(0.005),                                                             bias_regularizer=regularizers.l2(0.005)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, activation='relu',                                kernel_regularizer=regularizers.l2(0.001),                                bias_regularizer=regularizers.l2(0.001),))
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Dense(8, activation='relu',                                kernel_regularizer=regularizers.l2(0.001),                                bias_regularizer=regularizers.l2(0.001),))
model.add(tf.keras.layers.Dropout(0.4))


model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
                               



model.summary()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(1e-3),metrics=[tf.keras.metrics.BinaryAccuracy()])


# %%
tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)


# %%
epochs = 12
# Fit the model using the train and test datasets.
#history = model.fit(x_train, train_labels,validation_data= (x_test,test_labels),epochs=epochs )
history = model.fit(train_ds.shuffle(5000).batch(1024),
                    epochs= epochs ,
                    validation_data=valid_ds.batch(1024),
                    verbose=1)


# %%
history.history


# %%
plt.plot(history.history['loss'], label=' training data')
plt.plot(history.history['val_loss'], label='validation data')
plt.title('Loss for Text Classification')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()


# %%
plt.plot(history.history['binary_accuracy'], label=' training data')
plt.plot(history.history['val_binary_accuracy'], label='validation data')
plt.title('Accuracy for Text Classification')
plt.ylabel('Accuracy value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()


# %%
model.save('C:\\Users\\vikar\\Desktop\\Viraj Karkar\\Technocolab\\Supervised ML Task-1\\savedTFLSTMModel\\tf_lstmmodel.h5') 
json_string = tokenizer.to_json()


# %%
import json
with open('C:\\Users\\vikar\\Desktop\\Viraj Karkar\\Technocolab\\Supervised ML Task-1\\savedTFLSTMModel\\tokenizer.json', 'w') as outfile:
    json.dump(json_string, outfile)


# %%
valid_predict= model.predict(x_valid)


# %%
print(valid_predict[:10])


# %%
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, thresholds = sklearn.metrics.roc_curve(labels, predictions)
    plt.plot(fp, tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives Rate')
    plt.ylabel('True positives Rate')
    plt.xlim([-0.03, 1.0])
    plt.ylim([0.0, 1.03])
    plt.grid(True)
    thresholdsLength = len(thresholds)
    thresholds_every = 1000
    colorMap = plt.get_cmap('jet', thresholdsLength)
    for i in range(0, thresholdsLength, thresholds_every):
        threshold_value_with_max_four_decimals = str(thresholds[i])[:5]
        plt.text(fp[i] - 0.03, tp[i] + 0.001, threshold_value_with_max_four_decimals, fontdict={'size': 15}, color=colorMap(i/thresholdsLength));

    ax = plt.gca()
    ax.set_aspect('equal')


# %%
mpl.rcParams['figure.figsize'] = (16, 16)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# %%

plot_roc("Valid Baseline", valid_labels, valid_predict, color=colors[0], linestyle='--')
plt.legend(loc='lower right')


# %%
new_model = tf.keras.models.load_model("C:\\Users\\vikar\\Desktop\\Viraj Karkar\\Technocolab\\Supervised ML Task-1\\savedTFLSTMModel\\tf_lstmmodel.h5")
new_model.summary()


# %%
with open('C:\\Users\\vikar\\Desktop\\Viraj Karkar\\Technocolab\\Supervised ML Task-1\\savedTFLSTMModel\\tokenizer.json') as json_file:
    json_string = json.load(json_file)
tokenizer1 = tf.keras.preprocessing.text.tokenizer_from_json(json_string)


# %%
x_test  = np.array( tokenizer.texts_to_sequences(test_data['reviews_text'].tolist()) )
x_test = pad_sequences(x_test, padding='post', maxlen=100)


# %%
# Generate predictions (probabilities -- the output of the last layer)
# on test  data using `predict`
print("Generate predictions for all samples")
predictions = new_model.predict(x_test)


# %%
test_data['pred_sentiment']= predictions
test_data['pred_sentiment'] = np.where((test_data.pred_sentiment >= 0.78),1,test_data.pred_sentiment)
test_data['pred_sentiment'] = np.where((test_data.pred_sentiment < 0.78),0,test_data.pred_sentiment)


# %%
labels = [0, 1]
    
print(classification_report(test_data['sentiment'].tolist(),test_data['pred_sentiment'].tolist(),labels=labels))    


