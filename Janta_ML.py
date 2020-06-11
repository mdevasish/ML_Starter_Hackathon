#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt


df_train = pd.read_csv('train_ML.csv')
df_test = pd.read_csv('test_ML.csv')
df_test['is_pass'] = -1
df = pd.concat([df_train,df_test],ignore_index = True)
df


# In[2]:


df.info()


# In[3]:


# Verify few assumptions

df['new'] = df['program_id'].apply(lambda x: x[0])
df['new_id'] = df['id'].apply(lambda x: int(str(x).split('_')[1]))
print(df['new'].equals(df['program_type']))
print(df['new_id'].equals(df['test_id']))
df = df.drop(['new','new_id'],axis = 1)


# In[4]:


# Mapping between test_id and program_id
prg_list = set(df['program_id'].unique())
prg_map_testid = dict()
for each in tqdm(prg_list):
    x=df[df['program_id']==each]
    prg_map_testid[each] = set(x['test_id'].unique())
prg_map_testid


# In[5]:


df['is_handicapped'] = df['is_handicapped'].replace('N',0).replace('Y',1)
df['test_type'] = df['test_type'].replace('offline',0).replace('online',1)
df['gender'] = df['gender'].replace('M',1).replace('F',0)
df['education'] = df['education'].replace('Matriculation',1).replace('High School Diploma',2).replace('Bachelors',3).replace('Masters',4).replace('No Qualification',0)

df['age'] = df['age'].fillna(-99)
df['trainee_engagement_rating'] = df['trainee_engagement_rating'].fillna(0)
#df['city_tier'] = df['city_tier'].astype('category') # Verify if this needs to be chaged to dummy variables
df


# In[6]:


protype_dict = df_train.groupby(['program_type'])['is_pass'].mean().to_dict()
protype_dict


# In[7]:


df['program_type'] = df['program_type'].replace('S',protype_dict['S']).replace('T',protype_dict['T']).replace('U',protype_dict['U']).replace('V',protype_dict['V']).replace('Z',protype_dict['Z']).replace('X',protype_dict['X']).replace('Y',protype_dict['Y'])
df['program_type']


# In[8]:


dummies = pd.get_dummies(df['difficulty_level'])
dummies = dummies.drop(['vary hard'],axis = 1)
edu = pd.get_dummies(df['education'])
df = pd.concat([df,dummies],axis = 1)
df = df.drop(['difficulty_level','trainee_id'],axis = 1)
df


# In[9]:


mylist = []
for index, row in tqdm(df.iterrows()):
    mylist.append(str(str(row['program_id'])+'_'+str(row['test_id'])))


# In[10]:


x = pd.Series(mylist)
df = pd.concat([df,x],axis = 1)
df


# In[11]:


df.columns = ['id','program_id','program_type','program_duration','test_id','test_type','gender','education','city_tier','age',
'total_programs_enrolled','is_handicapped','trainee_engagement_rating','is_pass','easy','hard','intermediate','final_id']
df = df.drop(['program_id','test_id'],axis = 1)


# In[12]:


code_dict = df[df['is_pass']!= -1].groupby('final_id')['is_pass'].mean().to_dict()


# In[13]:


df['final_id'] = df['final_id'].map(code_dict)


# In[14]:


df = df[['id', 'program_type', 'program_duration', 'test_type', 'gender',
       'education', 'city_tier', 'age', 'total_programs_enrolled',
       'is_handicapped', 'trainee_engagement_rating','easy',
       'hard', 'intermediate', 'final_id','is_pass']]


# In[15]:


pr_df_train = df[df['is_pass'] != -1].drop('id',axis = 1)
pr_df_validate = df[df['is_pass'] == -1].drop(['id','is_pass'],axis = 1)


# In[16]:


m,n=pr_df_train.shape
m,n


# In[17]:


X = pr_df_train.iloc[:,0:n-1].values
y = pr_df_train.iloc[:,-1].values
X,y


# In[18]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)


# In[19]:


X_train.shape , X_test.shape, y_train.shape, y_test.shape


# In[20]:


clf = RandomForestClassifier(random_state = 41)
clf.fit(X_train,y_train)
print('Accuracy on train:',clf.score(X_train,y_train))
print('Accuracy on test:',clf.score(X_test,y_test))


# ## Feature Importance Plot

# In[37]:


fimp = pd.DataFrame(zip(pr_df_train.drop('is_pass',axis = 1).columns,clf.feature_importances_))
fimp.columns = ['features','Score']

sfimp =fimp.sort_values(by = 'Score')

plt.barh(sfimp['features'],sfimp['Score'])


# ## Random forest implmentation

# In[38]:


with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=200,random_state = 41)
    clf.fit(X_train,y_train)

    print('Accuracy on train:',clf.score(X_train,y_train))
    print('Accuracy on test:',clf.score(X_test,y_test))

    X_val = pr_df_validate.values
    y_pred = clf.predict(X_val)
    pd.DataFrame(list(y_pred)).to_csv('Output.csv')

    print('Feature Importance:', clf.feature_importances_)

    mlflow.log_param('random state', 41)
    mlflow.log_param('n_estimators', 200)
    mlflow.log_metric('accuracy', clf.score(X_test,y_test))


# ## Random forest Fine Tuning

# In[39]:


parameters = [{'criterion' : ['gini','entropy'],
               'max_depth' : [5,6,7],
               'n_estimators' : [100,125,150,175,200]}]

grid_search = GridSearchCV(estimator = clf,
                           param_grid = parameters,
                           cv = StratifiedKFold(n_splits = 5, shuffle = True),
                           scoring = 'accuracy',
                           n_jobs = -1
)
grid = grid_search.fit(X_train,y_train)
grid.best_params_, grid.best_score_


# In[40]:


with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100,random_state = 41,max_depth = 7,criterion = 'gini')
    clf.fit(X_train,y_train)
    
    print('Accuracy on train:',clf.score(X_train,y_train))
    print('Accuracy on test:',clf.score(X_test,y_test))
    
    y_pred = clf.predict(X_val)
    pd.DataFrame(list(y_pred)).to_csv('Output_1.csv')
    
    print('Feature Importance:', clf.feature_importances_)
    
    mlflow.log_param('random state', 41)
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('max_depth', 7)
    mlflow.log_param('criterion', 'gini')
    mlflow.log_metric('accuracy', clf.score(X_test,y_test))


# In[41]:


fimp = pd.DataFrame(zip(pr_df_train.drop('is_pass',axis = 1).columns,clf.feature_importances_))
fimp.columns = ['features','Score']

sfimp =fimp.sort_values(by = 'Score')

plt.barh(sfimp['features'],sfimp['Score'])


# ## AdaBoost Implementation

# In[42]:


with mlflow.start_run():
    clf = AdaBoostClassifier(random_state = 41)
    clf.fit(X_train,y_train)

    print('Accuracy on train:',clf.score(X_train,y_train))
    print('Accuracy on test:',clf.score(X_test,y_test))
    
    y_pred = clf.predict(X_val)
    pd.DataFrame(list(y_pred)).to_csv('Adaboost.csv')
    
    


# In[43]:


mlflow.log_param('random state', 41)
mlflow.log_param('n_estimators', 200)
mlflow.log_metric('accuracy', clf.score(X_test,y_test))


# In[44]:


mlflow.end_run()


# In[45]:


fimp = pd.DataFrame(zip(pr_df_train.drop('is_pass',axis = 1).columns,clf.feature_importances_))
fimp.columns = ['features','Score']

sfimp =fimp.sort_values(by = 'Score')

plt.barh(sfimp['features'],sfimp['Score'])


# In[46]:


parameters = [{'learning_rate' : [0.2,0.4,0.6,0.75,1],
               'n_estimators' : [150,200,225,250]}]
grid_search = GridSearchCV(estimator = clf,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1,
                           cv = StratifiedKFold(n_splits = 5, shuffle = True) 
)
grid = grid_search.fit(X_train,y_train)
grid.best_params_, grid.best_score_


# In[48]:


with mlflow.start_run():
    clf = AdaBoostClassifier(n_estimators = 200,random_state = 41,learning_rate = 1)
    clf.fit(X_train,y_train)
    
    print('Accuracy on train:',clf.score(X_train,y_train))
    print('Accuracy on test:',clf.score(X_test,y_test))
    
    y_pred = clf.predict(X_val)
    pd.DataFrame(list(y_pred)).to_csv('Adaboost_GCV.csv')
    
    mlflow.log_param('random state', 41)
    mlflow.log_param('n_estimators', 200)
    mlflow.log_metric('accuracy', clf.score(X_test,y_test))

fimp = pd.DataFrame(zip(pr_df_train.drop('is_pass',axis = 1).columns,clf.feature_importances_))
fimp.columns = ['features','Score']

sfimp =fimp.sort_values(by = 'Score')

plt.barh(sfimp['features'],sfimp['Score'])


# ## XGBoosting

# In[51]:


with mlflow.start_run():
    xgclf = XGBClassifier()
    xgclf.fit(X_train,y_train)
    
    print('Accuracy on train:',xgclf.score(X_train,y_train))
    print('Accuracy on test:',xgclf.score(X_test,y_test))
    
    y_pred = xgclf.predict(X_val)
    pd.DataFrame(list(y_pred)).to_csv('XGBoost.csv')
    
    mlflow.log_param('random state', 41)
    mlflow.log_param('n_estimators', 200)
    mlflow.log_metric('accuracy', xgclf.score(X_test,y_test))
    
fimp = pd.DataFrame(zip(pr_df_train.drop('is_pass',axis = 1).columns,xgclf.feature_importances_))
fimp.columns = ['features','Score']

sfimp =fimp.sort_values(by = 'Score')

plt.barh(sfimp['features'],sfimp['Score'])


# ## Tuning

# In[50]:


parameters = [{'learning_rate' : [0.4,0.6,0.75],
               'n_estimators' : [60,75,80],
               'max_depth' : [2,3,5]}]
grid_search = GridSearchCV(estimator = xgclf,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1,
                           cv = StratifiedKFold(n_splits = 4, shuffle = True) 
)
grid = grid_search.fit(X_train,y_train)
grid.best_params_, grid.best_score_


# In[61]:


with mlflow.start_run():
    xgclf = XGBClassifier(max_depth = 5, n_estimators = 250)
    xgclf.fit(X_train,y_train)
    
    print('Accuracy on train:',xgclf.score(X_train,y_train))
    print('Accuracy on test:',xgclf.score(X_test,y_test))
    
    y_pred = xgclf.predict(X_val)
    pd.DataFrame(list(y_pred)).to_csv('XGBoost_GCV_2.csv')
    
    mlflow.log_param('random state', 41)
    mlflow.log_param('n_estimators', 200)
    mlflow.log_metric('accuracy', clf.score(X_test,y_test))
    
fimp = pd.DataFrame(zip(pr_df_train.drop('is_pass',axis = 1).columns,clf.feature_importances_))
fimp.columns = ['features','Score']

sfimp =fimp.sort_values(by = 'Score')

plt.barh(sfimp['features'],sfimp['Score'])

#Accuracy on train: 0.7521776493105738
#Accuracy on test: 0.729915698336751

#Accuracy on train: 0.7627827038006327
#Accuracy on test: 0.7300979722032354


# In[55]:


XGBClassifier()


# In[53]:


clf = ExtraTreesClassifier()
clf.fit(X_train,y_train)
print('Accuracy on train:',clf.score(X_train,y_train))
print('Accuracy on test:',clf.score(X_test,y_test))
    
y_pred = clf.predict(X_val)
pd.DataFrame(list(y_pred)).to_csv('Extra_trees.csv')

fimp = pd.DataFrame(zip(pr_df_train.drop('is_pass',axis = 1).columns,clf.feature_importances_))
fimp.columns = ['features','Score']

sfimp =fimp.sort_values(by = 'Score')

plt.barh(sfimp['features'],sfimp['Score'])


# In[42]:


ExtraTreesClassifier()


# In[50]:


parameters = [{'max_features' : [14,15],
               'n_estimators' : [140,155,160],
               'max_depth' : [9,10,11]}]
grid_search = GridSearchCV(estimator = clf,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1,
                           cv = StratifiedKFold(n_splits = 5, shuffle = True) 
)
grid = grid_search.fit(X_train,y_train)
grid.best_params_, grid.best_score_


# In[51]:


clf = ExtraTreesClassifier(max_features = 14,n_estimators = 160,max_depth = 10)
clf.fit(X_train,y_train)
print('Accuracy on train:',clf.score(X_train,y_train))
print('Accuracy on test:',clf.score(X_test,y_test))

y_pred = clf.predict(X_val)

pd.DataFrame(list(y_pred)).to_csv('Extra_trees_1.csv')

fimp = pd.DataFrame(zip(pr_df_train.drop('is_pass',axis = 1).columns,clf.feature_importances_))
fimp.columns = ['features','Score']

sfimp =fimp.sort_values(by = 'Score')

plt.barh(sfimp['features'],sfimp['Score'])


# In[52]:


y_pred_proba = clf.predict_proba(X_val)
y_pred_proba


# In[56]:


pd.DataFrame(list(y_pred_proba),columns = ['Extra_P_0','Extra_P_1'])


# In[62]:


mlflow.search_runs()

