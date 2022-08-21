#!/usr/bin/env python
# coding: utf-8

# ## Music Genre Classificaiton:

# #### Importing required Libraries

# In[1]:


#for file reading and data analysis
import pandas as pd

#normalization and standarization
from sklearn.preprocessing import StandardScaler

#data visualization
import matplotlib.pyplot as plt

#for principal component analysis
from sklearn.decomposition import PCA

#for mathematical operations
import numpy as np

#for spliting data and decision tree model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#for logistic regression model
from sklearn.linear_model import LogisticRegression

#for comparison between decision tree and logistic regression
from sklearn.metrics import classification_report

#for cross validation
from sklearn.model_selection import KFold, cross_val_score

#for avoiding warnings
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ### Loading data:
# We have two resources to load data from. One is csv file and other one is json fiel.

# In[2]:


songs = pd.read_csv("fma-rock-vs-hiphop.csv")
echonest_metrics = pd.read_json("echonest-metrics.json", precise_float=True)


# In[3]:


songs.head(5)


# In[4]:


echonest_metrics.head()


# Now we will merge the both datasets into one on the track id, as both dataset has track_id column.

# In[5]:


data = echonest_metrics.merge(songs[["track_id", "genre_top"]], on="track_id")


# In[6]:


data.head(10)


# In[7]:


data.tail(10)


# In[8]:


print(data.shape)


# We have total of 4802 recodrs with 11 features(columns).

# In[9]:


print(data.info())


# In[10]:


data.describe().T


# The above table shows the descriptive statistics of our data.

# In[11]:


data.isnull().sum()


# Our data is clean, as there is no missing values.

# ## Pair-wise relationship between continous variables
#     Correlation is a statistical technique that shows how two variables are related.
#     We do this to avoid those varibales which has strong correlations.
#     We will use the pandas builtin method corr() Compute pairwise correlation of columns, excluding NA/null values.

# In[12]:


corr_matrix = data.corr()
corr_matrix

    The above correlation matrix shows that the relationship can be either positive or negative.As shown in the matrix there is not strong correlations between any two variabels. Every varibale has a strong correlation with itself. 
    
# ### Normalization:
#     Normalization refers to rescaling real-valued numeric attributes into a 0 to 1 range. Data normalization is used in machine learning to make model training less sensitive to the scale of features.

# First we will separate our features and labels, so we can apply normalization to our features only.

# In[ ]:





# In[13]:


features = data.drop(['genre_top','track_id'], axis = 1)
labels = data['genre_top']


#     Now we will scale the features through StandardScalar() from scikit learn.

# In[14]:


scaler = StandardScaler()
scaled_train_X = scaler.fit_transform(features)


# ### PCA:
#     Principal Component Analysis is an unsupervised learning algorithm that is used for the dimensionality reduction in machine learning. It is a statistical process that converts the observations of correlated features into a set of linearly uncorrelated features with the help of orthogonal transformation.
#     Now as we have preprocessed our data, we will use PCA to reduce the dimensions of our data.

# In[15]:


#create pca object
pca = PCA()

#fit the scaled data to pca object
pca.fit(scaled_train_X)

#now take the explained variance ratios of all variables through pca
variance_exp_ratios = pca.explained_variance_ratio_

#plot the explained variance ratio for visualization
fig, ax = plt.subplots()
ax.bar(range(pca.n_components_),variance_exp_ratios)
ax.set_xlabel('Principal Component #')


#     Now we will calcluate the cummulative explained variance, for futher PCA analysis.

# In[16]:


#calculate the cummulative explained variance
cumm_explained_var = np.cumsum(variance_exp_ratios)

#plot the cummulative explained varriance
fig, ax = plt.subplots()
ax.plot(cumm_explained_var)
ax.axhline(y=0.9, linestyle='--')

#indices where cummulative explained variance exceeds 0.9
n_components = ((np.where(cumm_explained_var> 0.9))[0][0]) 


#pca on the choosen components
pca = PCA(n_components, random_state=10)
pca.fit(scaled_train_X)
pca_projection = pca.transform(scaled_train_X)


# ## Train Decision Tree Model:
#     Now it's the time to train our first model Decision Tree.

# In[17]:


#split the data into training and test data
train_x, test_x, train_y, test_y = train_test_split(pca_projection,labels,random_state=10)

#decision tree model
model_1 = DecisionTreeClassifier(random_state=10)

#train the model with x_train and y_train
model_1.fit(train_x,train_y)

#make predictions with test data
preds_1 = model_1.predict(test_x) 


# ## Logistic Regression:
#     Now we will train our second model with the same data.

# In[18]:


#logistic regression model
model_2 = LogisticRegression(random_state=10)

#train the model with training data
model_2.fit(train_x, train_y)

#make predictions with test data
preds_2 = model_2.predict(test_x)


# ## Model Comparison:
#     It's a good idea to train your data with more than one model and to make comparison which one did the best.

# In[19]:


#make a classification report for decision tree model (model_1)
decision_tree_report = classification_report(test_y, preds_1)

#make a classification report for the logistic regression model(model_2)
log_regression_report = classification_report(test_y, preds_2)


# In[20]:


#let's print the reports
print("##################################################")
print("Decision Tree Model:\n", decision_tree_report)
print("###################################################")
print("Logistic Regression Model:\n", log_regression_report)


#     The given comparison shows that our both model did pretty well job by classifying the data into rock and hip hop music genre. But to take a close look at our results, we can see that our model classify the Rock music very accurately than the hip hop music. The reason is that for Rock music we have enough data points as compared to the Hip-Hop music genre. It means that our data is not balance. So, let's take care of it.

# ## Balance the dataset.
#     First thing first, let's separate our Rock music labels and Hip-Hop music labels from the dataset.

# In[21]:


hip_hop_only = data.loc[data['genre_top']=="Hip-Hop"]
rock_only = data.loc[data['genre_top']=="Rock"]


# In[22]:


hip_hop_only.head()


# In[24]:


rock_only.head()


# In[25]:


#make rock as much as hip-hop
rock_only = rock_only.sample(n=len(hip_hop_only), random_state=10)


# In[26]:


rock_only.shape


# In[28]:


hip_hop_only.shape


# In[29]:


#combine both
balanced_data = pd.concat([rock_only,hip_hop_only])


# In[30]:


balanced_data.head()


# In[31]:


balanced_data.tail()


# In[32]:


balanced_data.shape


# In[34]:


#separate features and labels for balanced data
features = balanced_data.drop(['genre_top', 'track_id'], axis=1)
labels = balanced_data['genre_top']


# In[36]:


#pca projection for the balanced data
pca_projection = pca.fit_transform(scaler.fit_transform(features))


# In[37]:


#split the balanced dataset into train and test sets
train_x, test_x, train_y, test_y = train_test_split(pca_projection, labels, random_state=10)


# ### Train the models with balanced datasets

# In[38]:


#train the tree model
tree_model = DecisionTreeClassifier(random_state=10)
tree_model.fit(train_x,train_y)
tree_preds = tree_model.predict(test_x)

#train the logistic regression model
log_model = LogisticRegression(random_state=10)
log_model.fit(train_x,train_y)
log_preds = log_model.predict(test_x)


# ## Compare the results

# In[40]:


print("Decison Tree Model: \n", classification_report(test_y,tree_preds))
print("Logistic Regression Model:\n", classification_report(test_y, log_preds))


# ### Look!!!!!!!!!
#     We got almost same accuracy for both Rock and Hip-Hop music.

# ## Cross Validation:
#     Cross-validation is a technique for evaluating ML models by training several ML models on subsets of the available input data and evaluating them on the complementary subset of the data. Use cross-validation to detect overfitting, ie, failing to generalize a pattern.We will use the KFold technique for cross validation.

# In[43]:


#set up KFold for cross validation
kfold = KFold(n_splits=5)

tree_model = DecisionTreeClassifier(random_state=10)
log_model = LogisticRegression(random_state=10)

#train the models with cross validation
tree_model_score = cross_val_score(tree_model,pca_projection,labels,cv=kfold)
logit_model_score = cross_val_score(log_model,pca_projection,labels,cv=kfold)


# ### Display the results:

# In[45]:


print("Decision Tree:", np.mean(tree_model_score))
print("\nLogistic Regression:", np.mean(logit_model_score))


# In[ ]:




