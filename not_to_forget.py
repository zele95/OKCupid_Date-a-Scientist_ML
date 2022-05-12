# %%
import re
from preprocessing import preprocess_text
import pandas as pd

df = pd.read_csv('profiles.csv')
df_e = df.loc[:,'essay0':'essay9']

df_e.fillna('Nan',inplace = True)

df_e = df_e.iloc[4000:6000,:].applymap(lambda x: preprocess_text(x))
# print(df_e.iloc[5849])

string5849 = 'may 2012 update:<br /><br />oh wow this profile could use an updating! give me a minute. it''llbe a better reflection of the current trip i''m on.<br /><br />on to the profile....<br />____________________________________________________<br />visionary, driven, compassionate, humanist, creative, stoic,humble, all describe me. i''ve always been pretty humble but i alsolove who i am. never full of myself but often joke that i am forcomic relief. part introvert, part extrovert. all honesty.<br /><br />labels that might be accurate: dj, counter-culture, burner, tourist/ world citizen, activist, office schlep, architect of internets,renegade, sweet heart. i''m a lot of things but definitely not onedimensional.<br /><br />i''m passionate about music and politics. music i find much morefulfilling, interesting, and amazing. if i had to pick between acareer in social justice &amp; music, music would win by a hugemargin. i will always be fighting in some way for causes i believein.<br /><br />there''s more of course, but it''s good to leave some mystery for afuture meeting.'
cleaned = preprocess_text(string5849)
print(cleaned)
# %%
# df_essay.zodiac.replace({'leo':0,
#                         'gemini':1,
#                         'libra':2,
#                         'cancer':3,
#                         'virgo':4,
#                         'taurus':5,
#                         'scorpio':6,
#                         'aries':7,
#                         'pisces':8,
#                         'sagittarius':9,
#                         'aquarius':10,
#                         'capricorn':11},inplace = True)

sklearn.preprocessing.LabelEncoder
# %%
training_data = None
test_data = None

print(X_train.shape)
# training_labels = training_labels.head(50)
# X_train = X_train.head(50)

for column in list(X_train.loc[:,'essay0':'essay9'].columns):
# bag of words
    cv = CountVectorizer(stop_words = 'english')
    training_vector = pd.DataFrame(cv.fit_transform(X_train[column]).todense(),columns=cv.get_feature_names_out())
    test_vector = pd.DataFrame(cv.transform(X_test[column]).todense(),columns=cv.get_feature_names_out())

    training_data = pd.concat([training_data,training_vector],axis = 1) if training_data is not None else training_vector
    test_data = pd.concat([test_data,test_vector],axis = 1) if test_data is not None else test_vector

training_data.index = X_train.index
test_data.index = X_test.index
print(training_data.shape)

# other columns
print(X_train[['smokes','drinks','drugs']].shape)


training_data = pd.concat([training_data, X_train[['smokes','drinks','drugs']]],axis = 1) 
print(training_data.shape)
test_data = pd.concat([test_data, X_test[['smokes','drinks','drugs']]],axis = 1) 

# Naive Bayes Classifier
zodiac_classifier = MultinomialNB()

zodiac_classifier.fit(training_data, training_labels)

predictions = zodiac_classifier.predict(test_data)
print(predictions)
print(f'Naive Bayes model score: {zodiac_classifier.score(test_data,y_test)}')