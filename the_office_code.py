# IMPORTING DATA + PACKAGES

# Importing all necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from matplotlib import ticker
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/Users/lukebenson/Downloads/theoffice_withwriter_v3.csv', encoding = "ISO-8859-1")


# the distribution of lines for the main 12 characters
character_names = df.character.value_counts().index.tolist()[:12]
character_lines = df.character.value_counts().tolist()[:12]

# let's choose only the most popular characters, and label everyone else as 'Other':
characters = ['Michael', 'Dwight', 'Jim', 'Pam', 'Andy', 'Angela',
              'Kevin', 'Erin', 'Oscar', 'Ryan', 'Darryl', 'Phyllis']
others = [c for c in df.character.value_counts().index.tolist() if c not in characters]
df.loc[df.character.isin(others), 'character'] = 'Other'


# new variable: indicating who the previous speaker was for each line. this takes a couple steps:
# 1. finding the starting line of each episode
season_episode = np.array(100*df['season']+df['episode'].values.tolist())
season_episode_moved = np.insert(season_episode[:-1], 0, 0, axis=0)
new_episode = np.array(season_episode != season_episode_moved)
# 2. creating a variable to indicate who spoke previously
line_speakers = np.array(df.character.values.tolist())
previous_speakers = np.insert(line_speakers[:-1], 0, 'Start', axis=0)
previous_speakers[new_episode] = 'Start'
df['previous_speaker'] = previous_speakers

# Remove NULL lines (ones that only have direction note)
df1 = df.loc[~pd.isnull(df.text)].copy()

# include lines from only the top 12 characters (we can't possibly predict characters with only a few lines)
df2 = df1.loc[df1.character.isin(characters)].copy()

# replace all numbers with a placeholder 'numeric',
# since the actual number itself is insignificant and the word 'numeric' does not appear anywhere in the script
df2['text'] = [re.sub(r'\w*\d\w*', 'numeric', line).strip() for line in df2['text']]

# sub out ... for the word "ellipses" so we can clean words with other repeating characters more easily
df2['text'] = [line.replace('...', ' ellipses') for line in df2['text']]

# replace dashes with 'dash'
df2['text'] = [line.replace('---', ' dash') for line in df2['text']]

# remove gibberish words like aaaaaah and aaagggh and replace them with a 'gibberish' placeholder
df2['text'] = [re.sub(r'[a-z]*(.)\1{2,}[a-z]*', 'gibberish', line.lower()) for line in df2['text']]
df2['text'] = [re.sub(r'\w*gibberish\w*', 'gibberish', line) for line in df2['text']]


#ADDING TEXT FEATURES

# how many words does the line contain?
df2['line_length'] = df2['text'].str.split().str.len()

# does the line contain a number?
df2['numeric'] = df2['text'].str.contains('numeric')

# does the line contain a question?
df2['question'] = df2['text'].str.contains("\?")

# does the line contain an exclamation point?
df2['exclamation'] = df2['text'].str.contains("\!")

# does the line contain director's notes?
df2['direction'] = df2['text_w_direction'].str.contains("\[")

# does the line contain ellipses?
df2['ellipses'] = df2['text'].str.contains("ellipses")

# does the line contain a dash mark?
df2['dash'] = df2['text'].str.contains("dash")

# does the line contain some kind of gibberish?
df2['gibberish'] = df2['text'].str.contains("gibberish")

# this is a built-in function from nltk to determine the sentiment score of a line
sid = SIA()
df2['sentiment'] = [sid.polarity_scores(line)['compound'] for line in df2['text']]


#TRAIN + TEST SET
# we subset our data into 8,000 test lines (~20% of our data) and ~34,000 training lines
# we will use the training lines to create our models, and then test our models on the test data
np.random.seed(1)
indices = np.array(df2.index)
np.random.shuffle(indices)
test_indices = indices[:8000]
train_indices = indices[8000:]

train_df2 = df2.loc[train_indices]
test_df2 = df2.loc[test_indices]


#UNIGRAM LANGUAGE MODEL

# creating a variable for the whole script
script = df2.text.to_numpy()

# stemming our words so words like 'abandon' and 'abandoned' are interpreted to be the same
ps = PorterStemmer()
stemmed_script = [[ps.stem(word) for word in s.split()] for s in script]
script = [" ".join(s) for s in stemmed_script]

# getting the total number of unigrams (words)
count_vector = CountVectorizer()
word_counts = count_vector.fit_transform(script)
all_script_words = count_vector.get_feature_names()

# data frame for the unigrams in each line
line_word_counts = pd.DataFrame(word_counts.toarray(), index=df2.index,columns=all_script_words)
line_word_counts.insert(0, 'office_character', df2.character)

# when we train, we don't want to use our test lines in our training data
# therefore, we will find the unigram and bigram distribution for each character using only the training data

# splitting into train and test data
train_lwc = line_word_counts.loc[train_indices]
test_lwc = line_word_counts.loc[test_indices]

# we want to train our unigram language model using just the training lines
character_word_counts = train_lwc.groupby('office_character').sum()

# we want to add Laplace smoothing: this code adds our vocab size to the denominator
V = len(character_word_counts.columns)
character_total_words = np.array(character_word_counts.sum(axis=1)) + V

# this code adds 1 to the numerator for Lapplace smoothing, and now we have P(unigram | character) for every combo
unigram_matrix = (character_word_counts+1).divide(character_total_words, axis =0)


#UNIGRAM PROBABILITIES FOR EACH LINE

# this will be necessary for matrix multiplication
del train_lwc['office_character']
del test_lwc['office_character']

# let's convert our probabilities into log probabilities so that we can add them
unigram_matrix_log_probs = np.log(unigram_matrix.T)

# we now have a data frame of unigrams in each line,
# and a data frame representing P(unigram | character) for each character built only off our training data.
# we can find the probabilities, P(line | character) for each line in the train set and test set
# by multiplying our P(unigram | character) data frame with our data frame of lines and unigram counts.
train_unigram_features = train_lwc.dot(unigram_matrix_log_probs)
test_unigram_features = test_lwc.dot(unigram_matrix_log_probs)

# renaming the columns
train_unigram_features.columns = ['unigram_' + str(col) for col in train_unigram_features.columns]
test_unigram_features.columns = ['unigram_' + str(col) for col in test_unigram_features.columns]

# remember, what we're looking for is P(character | line), so we can use these probabilities above
# as features in our logistic regression model!


#DESIGNING AND EVALUATING LOG REG MODELS

# we have a train set and a test set. for each model, we will train the model on the train set,
# and then evaluate its performance on the test set.
# this will give us some idea as to how well our model is generalizing to the data.
# our best model will be the one that performs the best on the test set.

# we can create our outputs first, since they will be the same for every model
y_train = np.array(train_df2['character'])
y_test = np.array(test_df2['character'])

#MODEL 1

# to establish a baseline, we will start by designing a model trained only on non-text features.
# this includes season, and previous line speaker
X_1 = pd.DataFrame({
    'season': df2.season
})

for name in (['Start'] + characters):
    column = 'previous_' + name
    X_1[column] = (df2['previous_speaker'] == name)

# separating our feature matrix into train and test
X_1_train = X_1.loc[train_indices]
X_1_test = X_1.loc[test_indices]

# to ease computation and interpretation of coefficient outputs, we need to standardize our data
# so that each variable has mean 0 and standard deviation 1
scaler = preprocessing.StandardScaler().fit(X_1_train)
X_1_train_scaled = scaler.transform(X_1_train)
X_1_test_scaled = scaler.transform(X_1_test)

# training our model ('newton-cg' proved to be the best and most efficient optimizer for our models)
log_reg_1 = LogisticRegression(multi_class = 'multinomial', solver='newton-cg', max_iter=1000)
log_reg_1.fit(X_1_train_scaled, y_train)

# predicting both our train and test lines
yhat_1_train = log_reg_1.predict(X_1_train_scaled)
yhat_1_test = log_reg_1.predict(X_1_test_scaled)

# we use a weighted f1 score because the distribution of lines over characters
# is far from uniform
print("Train Accuracy: %.4f | Test Accuracy: %.4f" % (accuracy_score(y_train, yhat_1_train),
                                                     accuracy_score(y_test, yhat_1_test)))
print("Train F1: %.4f | Test F1: %.4f" % (f1_score(y_train, yhat_1_train, average='weighted'),
                                         f1_score(y_test, yhat_1_test, average='weighted')))
# this can be our baseline: of we just know the season and who spoke before, this is the accuracy we get

# plotting a confusion matrix gives us some insight into where our model is going wrong
conf_mat_1 = confusion_matrix(y_test, yhat_1_test, characters)
# expressing as probabilities of the real character will be more interpretable
conf_mat_probs_1 = conf_mat_1/conf_mat_1.sum(axis=1)[:,None]

# ideally, we will be able to see the diagonals somewhat lit up
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat_probs_1)
plt.title('Character Predictions')
fig.colorbar(cax)
ax.set_xticklabels([''] + characters)
ax.set_yticklabels([''] + characters)
plt.xlabel('Predicted Character')
plt.ylabel('True Character')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# our model is actually getting Erin, a more minor character, sometimes.
# this may be true because she doesn't appear until later seasons, so our model knows
# not to predict her in an early season

#MODEL 2

# now let's use just the unigram probabilities themselves
# adding in the unigram features
X_2_train = train_unigram_features.copy()
X_2_test = test_unigram_features.copy()

# standardizing
scaler = preprocessing.StandardScaler().fit(X_2_train)
X_2_train_scaled = scaler.transform(X_2_train)
X_2_test_scaled = scaler.transform(X_2_test)

# training
log_reg_2 = LogisticRegression(multi_class = 'multinomial', solver='newton-cg', max_iter=1000)
log_reg_2.fit(X_2_train_scaled, y_train)

# predicting both our train and test lines
yhat_2_train = log_reg_2.predict(X_2_train_scaled)
yhat_2_test = log_reg_2.predict(X_2_test_scaled)

print("Train Accuracy: %.4f | Test Accuracy: %.4f" % (accuracy_score(y_train, yhat_2_train),
                                                     accuracy_score(y_test, yhat_2_test)))
print("Train F1: %.4f | Test F1: %.4f" % (f1_score(y_train, yhat_2_train, average='weighted'),
                                         f1_score(y_test, yhat_2_test, average='weighted')))
# does well on the training set, but poorly on the test set: a sign that it's overfitting

conf_mat_2 = confusion_matrix(y_test, yhat_2_test, characters)
conf_mat_probs_2 = conf_mat_2/conf_mat_2.sum(axis=1)[:,None]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat_probs_2)
plt.title('Character Predictions')
fig.colorbar(cax)
ax.set_xticklabels([''] + characters)
ax.set_yticklabels([''] + characters)
plt.xlabel('Predicted Character')
plt.ylabel('True Character')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# we're really just predicting a lot of Michael, which makes sense.
# Michael has the most lines and says the most words, so just using unigram probabilities means that
# we'll guess Michael the majority of the time

#MODEL 3

# now let's use our non-text features, and our text features outside of unigrams which we engineered
X_3 = pd.DataFrame({
    'season': df2.season,
    'line_length': df2.line_length,
    'question': df2.question,
    'exclamation': df2.exclamation,
    'direction': df2.direction,
    'ellipses': df2.ellipses,
    'dash': df2.dash,
    'gibberish': df2.gibberish,
    'sentiment': df2.sentiment
})

for name in (['Start'] + characters):
    column = 'previous_' + name
    X_3[column] = (df2['previous_speaker'] == name)

# train and test
X_3_train = X_3.loc[train_indices]
X_3_test = X_3.loc[test_indices]

# standardizing
scaler = preprocessing.StandardScaler().fit(X_3_train)
X_3_train_scaled = scaler.transform(X_3_train)
X_3_test_scaled = scaler.transform(X_3_test)

# training
log_reg_3 = LogisticRegression(multi_class = 'multinomial', solver='newton-cg', max_iter=1000)
log_reg_3.fit(X_3_train_scaled, y_train)

# predicting both our train and test lines
yhat_3_train = log_reg_3.predict(X_3_train_scaled)
yhat_3_test = log_reg_3.predict(X_3_test_scaled)

print("Train Accuracy: %.4f | Test Accuracy: %.4f" % (accuracy_score(y_train, yhat_3_train),
                                                     accuracy_score(y_test, yhat_3_test)))
print("Train F1: %.4f | Test F1: %.4f" % (f1_score(y_train, yhat_3_train, average='weighted'),
                                         f1_score(y_test, yhat_3_test, average='weighted')))
# perhaps evidence of underfitting, as both train and test accuracy are lower than desired

conf_mat_3 = confusion_matrix(y_test, yhat_3_test, characters)
conf_mat_probs_3 = conf_mat_3/conf_mat_3.sum(axis=1)[:,None]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat_probs_3)
plt.title('Character Predictions')
fig.colorbar(cax)
ax.set_xticklabels([''] + characters)
ax.set_yticklabels([''] + characters)
plt.xlabel('Predicted Character')
plt.ylabel('True Character')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#MODEL 4

# this is the big model: we will include our non-text features, our engineered text features, and
# our unigram features

# now let's use our non-text features, and our text features outside of unigrams which we engineered
X_4 = pd.DataFrame({
    'season': df2.season,
    'line_length': df2.line_length,
    'question': df2.question,
    'exclamation': df2.exclamation,
    'direction': df2.direction,
    'ellipses': df2.ellipses,
    'dash': df2.dash,
    'gibberish': df2.gibberish,
    'sentiment': df2.sentiment
})

for name in (['Start'] + characters):
    column = 'previous_' + name
    X_4[column] = (df2['previous_speaker'] == name)

# train and test
X_4_train = X_4.loc[train_indices]
X_4_test = X_4.loc[test_indices]

# train and test
X_4_train = X_4.loc[train_indices]
X_4_test = X_4.loc[test_indices]

# standardizing
scaler = preprocessing.StandardScaler().fit(X_4_train)
X_4_train_scaled = scaler.transform(X_4_train)
X_4_test_scaled = scaler.transform(X_4_test)

# training
log_reg_4 = LogisticRegression(multi_class = 'multinomial', solver='newton-cg', max_iter=1000)
log_reg_4.fit(X_4_train_scaled, y_train)

# predicting both our train and test lines
yhat_4_train = log_reg_4.predict(X_4_train_scaled)
yhat_4_test = log_reg_4.predict(X_4_test_scaled)

print("Train Accuracy: %.4f | Test Accuracy: %.4f" % (accuracy_score(y_train, yhat_4_train),
                                                     accuracy_score(y_test, yhat_4_test)))
print("Train F1: %.4f | Test F1: %.4f" % (f1_score(y_train, yhat_4_train, average='weighted'),
                                         f1_score(y_test, yhat_4_test, average='weighted')))
# this is the best model

conf_mat_4 = confusion_matrix(y_test, yhat_4_test, characters)
conf_mat_probs_4 = conf_mat_4/conf_mat_4.sum(axis=1)[:,None]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat_probs_4)
plt.title('Character Predictions')
fig.colorbar(cax)
ax.set_xticklabels([''] + characters)
ax.set_yticklabels([''] + characters)
plt.xlabel('Predicted Character')
plt.ylabel('True Character')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# we can see that this model also does the best at predicting characters outside of Michael.
# e.g. Andy is predicted to be himself most of the time and Erin is predicted to be herself
# a decent number of times as well.


#FEATURE IMPORTANCE

# average overall
fig = plt.figure(figsize = (12,4))
ax = fig.add_subplot(111)
ax.bar(list(X_4_train.columns), np.mean(np.abs(log_reg_4.coef_), axis=0))
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.title("Average Absolute Value of Weights", fontsize=18)

# example: Kevin
fig = plt.figure(figsize = (12,4))
ax = fig.add_subplot(111)
ax.bar(list(X_4_train.columns), np.mean(np.abs(log_reg_4.coef_), axis=0))
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.title("Average Absolute Value of Weights", fontsize=18)

fig = plt.figure(figsize = (16,6))
ax = fig.add_subplot(111)
ax.bar(list(X_3_train.columns), list(log_reg_3.coef_[6,:]))
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.title("Weights for " + sorted(characters)[6], fontsize=18)
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)