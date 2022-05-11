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