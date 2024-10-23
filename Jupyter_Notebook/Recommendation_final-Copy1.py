#!/usr/bin/env python
# coding: utf-8

# <h2>Anime Collaborative Recommendation System</h2>

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('/Users/owais/Jupyter Notebook/archive-2/rating.csv')


# In[3]:


df


# In[4]:


len(df['anime_id'].unique())


# In[5]:


len(df['user_id'].unique())


# In[6]:


11200-9480


# In[7]:


df.info()


# In[8]:


df.describe().T


# In[9]:


df = df.replace(np.nan, 0)


# In[10]:


# rating_counts = pd.DataFrame(df["anime_id"])
# rating_counts.reset_index(drop=True)
# rating_counts['anime_id'] = pd.DataFrame(df['anime_id'].value_counts())
rating_counts = pd.DataFrame(df['anime_id'].value_counts())


# In[11]:


# rare_movies = rating_counts[rating_counts["anime_id"] <= 1000].index
# common_movie = df[~df["anime_id"].isin(rare_movies)]
rating_counts.columns


# In[12]:


rare_movies = rating_counts[rating_counts["count"] <= 1000].index
common_movie = df[~df["anime_id"].isin(rare_movies)]


# In[13]:


common_movie


# In[14]:


cm = df[~df["anime_id"].isin(rare_movies)]


# In[15]:


cm.info()


# In[16]:


len(cm['anime_id'].unique())


# In[17]:


cm


# In[18]:


common_movie = common_movie.pivot_table(index = "user_id", columns = "anime_id")

common_movie.head()


# In[19]:


common_movie = common_movie.replace(np.nan, 0)


# In[20]:


common_movie


# In[21]:


# Calculate the non-zero counts for each row (axis=1)
non_zero_counts = common_movie.apply(lambda row: (row != 0).sum(), axis=1)

# Define a threshold (e.g., 10 non-zero values)
threshold = 10

# Select rows that meet the threshold
selected_rows = non_zero_counts[non_zero_counts >= threshold].index

# Create a new pivot table with selected rows
filtered_pivot_table = common_movie.loc[selected_rows]


# In[22]:


filtered_pivot_table


# In[23]:


filtered_pivot_table.info()


# In[24]:


# common_movie_reset['user_id'].unique()


# In[25]:


# len(common_movie_reset['user_id'].unique())


# In[26]:


my_ratings = np.array(common_movie.iloc[4])


# In[27]:


already_watched=pd.DataFrame(common_movie.iloc[4][common_movie.iloc[4] != 0])


# In[28]:


# common_movie.iloc[4][common_movie.iloc[4] != 0]


# In[29]:


already_watched.reset_index(inplace=True)
already_watched.set_index(['anime_id'], inplace=True)
already_watched.drop('level_0',axis=1) 


# In[30]:


aw=already_watched.index


# In[31]:


# (common_movie.iloc[1721]) #observation


# In[32]:


my_ratings


# <h3>Vector of favorite anime</h3>

# In[33]:


# list_movie = df["anime_id"]
list_movie = cm["anime_id"]
list_m=sorted(set(list_movie))


# In[34]:


list_movie


# In[35]:


len(list_m)


# In[36]:


list_m


# In[37]:


#example of anime favorite anime list
# anime_id_favorite = [32379,6,5,1,32438]
anime_id_favorite = [269,19815,1575,164,16498]
#example of vote of each anime
vote = [8,10,9,8,9] 
#creating the vector of favorit anime 
a = 1720
b = len(anime_id_favorite)
vector = []
for i in range(a):
    for j in range(b):
        if anime_id_favorite[j] == list_m[i]:
            vector.append(vote[j])
            break
        elif j==b-1:
            vector.append(0)


# In[38]:


# [269,19815,1575,164,16498]
aw_custom=pd.Index(anime_id_favorite)


# In[39]:


aw_custom


# In[40]:


vector = np.array(vector).T
len(vector)


# In[41]:


vector


# <h2>Cosine Similarity</h2>

# In[42]:


from scipy.spatial import distance


# In[43]:


# len(vector)


# In[44]:


len(common_movie.iloc[[2]].T)


# In[45]:


d = len(filtered_pivot_table)
score = []
for i in range(d):
    score.append(distance.cosine(np.squeeze(filtered_pivot_table.iloc[[i]]), vector))

# eg = np.squeeze(common_movie.iloc[[i]])
# eg.shape


# In[46]:


score


# In[47]:


len(score)


# In[48]:


type(score)


# In[49]:


# score.pop(4)


# In[50]:


score


# In[51]:


min(score)


# In[52]:


len(score)


# In[53]:


most_similar_user = score.index(min(score))


# In[54]:


most_similar_user


# In[55]:


filtered_pivot_table.iloc[53421]


# In[56]:


filtered_pivot_table.iloc[[53421]]


# In[57]:


distance.cosine(filtered_pivot_table.iloc[most_similar_user], vector)


# In[58]:


# most_similar_user+=1


# In[59]:


most_similar_user


# In[60]:


# score.pop(4071)


# In[61]:


min(score)


# In[62]:


common_movie.iloc[60042]


# In[63]:


common_movie_reset = common_movie.reset_index()


# In[64]:


common_movie_reset


# In[65]:


common_movie_reset.iloc[60043] #since we dropped the user we were taking as reference


# In[66]:


common_movie.iloc[60043]


# In[67]:


# # Replace 'row_index' with the index of the row you're interested in
# row_index = 5  # Change this to your desired row index

# # Select the row using .loc[] and then filter for non-zero values
# non_zero_values = common_movie.loc[row_index][common_movie.loc[row_index] != 0]


# In[68]:


my_ratings


# In[69]:


row_index = most_similar_user

non_zero_values = filtered_pivot_table.iloc[row_index][filtered_pivot_table.iloc[row_index] != 0]


# In[70]:


non_zero_values


# In[71]:


common_movie.loc[3]


# In[72]:


common_movie.iloc[most_similar_user].loc[common_movie.iloc[most_similar_user]!=0]
# common_movie.iloc[row_index][common_movie.iloc[row_index] != 0]


# In[73]:


common_movie.loc[60043]


# In[74]:


common_movie.iloc[60043]


# In[75]:


recm = pd.DataFrame(non_zero_values)
recm


# In[76]:


recm.info()


# In[77]:


recm.reset_index(inplace=True)


# In[78]:


recm


# In[79]:


recm.drop('level_0', axis=1, inplace=True)


# In[80]:


recm.rename(columns={64137: 'Ratings'}, inplace=True)


# In[81]:


recm.info()


# In[82]:


recm


# In[83]:


recm.set_index(['anime_id'], inplace=True)


# In[84]:


recm


# In[85]:


id=recm.loc[recm['Ratings']>=7].index
id


# In[86]:


anime = pd.read_csv('/Users/owais/Jupyter Notebook/archive-2/anime.csv')


# In[87]:


anime


# In[88]:


recommendations = anime[anime["anime_id"].isin(id)]


# In[89]:


recommendations


# In[90]:


recommendations[~recommendations["anime_id"].isin(aw_custom)]


# In[95]:


recommendations[recommendations["anime_id"].isin(aw_custom)]


# In[91]:


len(aw_custom)


# In[92]:


len(common_movie)


# In[93]:


anime


# In[94]:


anime[anime['name'].str.contains('kyojin',case=False)]

