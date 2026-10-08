
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb

import matplotlib.gridspec as gridspec

from scipy import stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, r2_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
        
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv(Path(__file__).with_name("cancer patient data sets.csv"))
print(df)
df.drop(columns=['index', 'Patient Id'], inplace=True)
print(df)
print(df.describe())
print(df.info())

print('Cancer Levels:', df['Level'].unique())

# Convert cancer level to numeric
df["Level"] = df["Level"].replace({
    'High': 2,
    'Medium': 1,
    'Low': 0
}).astype(int)

print('Cancer Levels:', df['Level'].unique())
print('Level datatype:', df['Level'].dtype)

plt.figure(figsize=(10, 4))

plt.boxplot(
    df['Age'],
    vert=False,
    patch_artist=True,
    boxprops=dict(facecolor='skyblue', linewidth=2),
    whiskerprops=dict(color='green', linewidth=3),
    medianprops=dict(color='red', linewidth=2)
)

plt.title('Distribution of data by age of patients')
plt.xticks(np.arange(10, max(df['Age']) + 1, 3))
plt.show()

import statsmodels.api as sm

data = df['Age'].dropna()

# QQ Plot
sm.qqplot(data, line='s')
plt.title("Data distribution for the 'Age' column")
plt.show()

# Correlation
df_corr = df.corr()

print(df_corr)

plt.figure(figsize=(20, 15))
sns.heatmap(
    df_corr,
    annot=True,
    cmap=plt.cm.PuBu
)
plt.title("Correlation Matrix")
plt.show()
print(sns.heatmap(df_corr, cmap='viridis'))
print('\n')
plt.figure(figsize=(20,15))
print(sns.heatmap(df.corr(), annot=True, cmap=plt.cm.PuBu))
plt.show()
print('\n')
sea = sns.FacetGrid(df, col = "Level", height = 5)
print(sea.map(sns.distplot, "Age", color="blue"))
sea = sns.FacetGrid(df, col = "Level", height = 5)
print(sea.map(sns.distplot, "Gender", color="purple"))

plt.figure(figsize = (15, 55))

for i in range(24):
    plt.subplot(16, 2, i+1)
    sns.distplot(df.iloc[:, i], color = 'red')
    plt.grid()

plt.figure(figsize = (15,7))
colors = ['red', 'yellow', 'green']
plt.title("Lung Cancer Chances ")
plt.pie(df['Level'].value_counts(), explode = (0.1, 0.02, 0.02), labels = ['High', 'Medium', 'Low'], autopct = "%1.2f%%", shadow = True, colors = colors)
plt.legend(title = "Lung Cancer Chances", loc = "lower left")

print(sns.displot(df['Level'], kde=True, color = 'red'))

dfviz = df.copy()
y = df.pop('Level')
x = df
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Results of prediction
results = pd.DataFrame(columns = ['model', 'f1_train', 'f1_test', 'r2_train', 'r2_test'])
print('X train shape: ', x_train.shape)
print('Y train shape: ', y_train.shape)
print('\nTest Shape\n')
print('X test shape: ', x_test.shape)
print('Y test shape: ', y_test.shape)

def performTest(y_pred):
    print("Test Data Metrics:")
    print("Precision : ", precision_score(y_test, y_pred, average = 'micro'))
    print("Recall : ", recall_score(y_test, y_pred, average = 'micro'))
    print("Accuracy : ", accuracy_score(y_test, y_pred))
    print("F1 Score : ", f1_score(y_test, y_pred, average = 'micro'))
    print("R2 Score : ", r2_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("\n", cm)
    print("\n")
    print("**"*27 + "\n" + " "* 16 + "Classification Report\n" + "**"*27)
    print(classification_report(y_test, y_pred))
    print("**"*27+"\n")

    cm = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Low', 'Medium', 'High'])
    
    cm.plot( cmap='plasma', ax=None, xticks_rotation='horizontal')

def performTrain(y_pred_train):
    print("Train Data Metrics:")
    print("Precision : ", precision_score(y_train, y_pred_train, average='micro'))
    print("Recall : ", recall_score(y_train, y_pred_train, average='micro'))
    print("Accuracy : ", accuracy_score(y_train, y_pred_train))
    print("F1 Score : ", f1_score(y_train, y_pred_train, average='micro'))
    print("R2 Score : ", r2_score(y_train, y_pred_train))
    print("\n")
    
from sklearn.model_selection import GridSearchCV
param_RF1 = {
    'n_estimators': 50, 
    'max_depth': 3,
    'min_samples_split': 3,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'max_samples': 0.8,
    'criterion': 'gini'    
}
param_RF = {
    'n_estimators': [20, 25, 50], 
    'max_depth': [2, 3],
    'min_samples_split': [3],  
    #'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [2, 3],
    'max_features': ['sqrt'],
    #'max_samples': [0.5, 0.6, 0.7]
    'max_samples': [0.4]
}

model_tuning = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_RF, cv=5)
model_tuning.fit(x_train, y_train)

best_params = model_tuning.best_params_
print("Best Parameters:", best_params)
model_rf = RandomForestClassifier(**best_params)
model_rf.fit(x_train, y_train)

train_predictions = model_rf.predict(x_train)
r2_train = r2_score(y_train, train_predictions)
f1_train = f1_score(y_train, train_predictions, average = 'micro')

# Test
test_predictions = model_rf.predict(x_test)
r2_test = r2_score(y_test, test_predictions)
f1_test = f1_score(y_test, test_predictions, average = 'micro')

#Result
performTrain(train_predictions)
performTest(test_predictions)

# Save
results.loc[0,'model'] = 'RandomForest Classifier'
results.loc[0,'f1_train'] = f1_train
results.loc[0,'f1_test'] = f1_test
results.loc[0,'r2_train'] = r2_train
results.loc[0,'r2_test'] = r2_test
results.loc[0,'short'] = 'RF'
score_model_rf = model_rf.score(x_test, y_test)


feature_names = dfviz.columns[0:23]
viz = dfviz.copy()
viz["Level"]=viz["Level"].values.astype(str)
print(viz.dtypes)
target_names = viz['Level'].unique().tolist()


params_XGB1 ={'n_estimators': 366,
                  'num_leaves': 10,
                  'max_depth': 9,
                 'lambda': 0.1444861779926268,
                  'subsample': 0.01,
                  'alpha': 2.603602561261043e-06,
                   'colsample_bytree': 1.0  }

params_XGB ={'n_estimators': [100, 200],
             'num_leaves': [2, 5],
             'max_depth': [3],         
             'subsample': [0.01],
             'learning_rate': [0.01],
             'objective': ['multi:softmax'],
             'num_class': [3]}

model_tuning = GridSearchCV(estimator=XGBClassifier(n_jobs=1, tree_method="hist"), param_grid=params_XGB, cv=3)
model_tuning.fit(x_train, y_train)

best_params = model_tuning.best_params_
print("Best Parameters:", best_params)
model_xgb = XGBClassifier(n_jobs=1, tree_method="hist", **best_params)
model_xgb.fit(x_train, y_train)


# Train
train_predictions = model_xgb.predict(x_train)
r2_train = r2_score(y_train, train_predictions)
f1_train = f1_score(y_train, train_predictions, average = 'micro')

# Test
test_predictions = model_xgb.predict(x_test)
r2_test = r2_score(y_test, test_predictions)
f1_test = f1_score(y_test, test_predictions, average = 'micro')

#Result
performTrain(train_predictions)
performTest(test_predictions)

# Save
results.loc[4,'model'] = 'XGB Classifier'
results.loc[4,'f1_train'] = f1_train
results.loc[4,'f1_test'] = f1_test
results.loc[4,'r2_train'] = r2_train
results.loc[4,'r2_test'] = r2_test
results.loc[4,'short'] = 'XGB'
score_model_xgb = model_xgb.score(x_test, y_test)

params_mlp1 ={'hidden_layer_sizes': 100,
            'random_state': 2,
            'alpha': 0.001,
            'activation': 'relu',
            'learning_rate_init': 0.001,
            'max_iter': 100,
             'batch_size': 32,
             'early_stopping':True,
             'validation_fraction': 0.1,
             'tol': 1e-4
            }

params_mlp2 ={'hidden_layer_sizes': [50, 75, 100],
             'random_state': [2],
             'alpha': [0.01, 0.001],
             'activation': ['relu', 'tanh'],
             'learning_rate_init': [0.001, 0.01],
             'max_iter': [100, 200, 300],
             'solver': ['adam', 'lbfgs'],
             'early_stopping': [True],
             'validation_fraction': [0.1, 0.2],
             
            }

params_mlp ={'hidden_layer_sizes': [50, 75, 100],
             'random_state': [2],
             'alpha': [0.001],
             'activation': ['relu', 'tanh'],
             'learning_rate_init': [0.001],
             'max_iter': [100, 200, 300],
             'solver': ['adam'],
             'early_stopping': [True],
             'validation_fraction': [0.1],
             
            }

model_tuning = GridSearchCV(estimator=MLPClassifier(), param_grid=params_mlp, cv=5)
model_tuning.fit(x_train, y_train)

best_params = model_tuning.best_params_
print("Best Parameters:", best_params)
model_mlp = MLPClassifier(**best_params)
model_mlp.fit(x_train, y_train)



colors = ['blue', 'green', 'red', 'purple', 'orange', 'c']
colors1 = ['c', 'lime', 'pink', 'magenta', 'yellow', 'cyan']

# Construction of a histogram
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = range(len(results['short']))

bar1 = plt.bar(index, results['f1_train'], bar_width,
                   label='f1_train ', color='red')
bar2 = plt.bar([i + bar_width for i in index], results['f1_test'], bar_width,
                label='f1_test', color='cyan')

#scatter1 = plt.scatter(index, results['f1_train'], 
#                   label='f1_train (dark color)',s=500, color=colors)
#scatter2 = plt.scatter([i + bar_width for i in index], results['f1_test'], 
#                label='f1_test (light color)', s=500, color=colors1)

plt.xlabel('Models')
plt.ylabel('f1_score')
plt.title('Results of the f1_train and f1_test evaluation by models')
plt.xticks([i + bar_width/2 for i in index], results['short'])
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

plt.tight_layout()
plt.show()

feature_importances_model_rf = pd.DataFrame(x_train.columns.delete(0))
feature_importances_model_rf.columns = ['feature']
feature_importances_model_rf["score_model_rf"] = pd.Series(model_rf.feature_importances_)
feature_importances_model_rf.sort_values(by='score_model_rf', ascending=False)

importances = model_rf.feature_importances_
feature_names = df.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='lime')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

