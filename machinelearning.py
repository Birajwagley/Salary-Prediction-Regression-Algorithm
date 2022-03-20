import pandas as pd
import matplotlib.pyplot as plt

# data preprocessing
df = pd.read_csv("survey_results_public.csv")
print(df.head())

df=df.rename({"WorkWeekHrs":"WorkingTime"},axis=1)

df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp","WorkingTime"]]
df = df.rename({"ConvertedComp": "Salary"}, axis=1)
print(df.head())

df = df[df["Salary"].notnull()]
print(df.head())

print(df.info())

df = df.dropna()
df.isnull().sum()

df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)
df.info()

print(df['Country'].value_counts())

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

country_map = shorten_categories(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)
print(df.Country.value_counts())

ig, ax = plt.subplots(1, 1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
print(plt.show())

df = df[df["Salary"] <= 250000]
df = df[df["Salary"] >= 10000]
df = df[df['Country'] != 'Other']

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
print(plt.show())

print(df["YearsCodePro"].unique())

def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

print(df["EdLevel"].unique())

def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

df['EdLevel'] = df['EdLevel'].apply(clean_education)

print(df["EdLevel"].unique())

print(df["WorkingTime"].value_counts())

df = df[df["WorkingTime"] <= 50]
df = df[df["WorkingTime"] >= 30]
df = df[df['WorkingTime'] != 'Other']

from sklearn.preprocessing import LabelEncoder

le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
print(df["EdLevel"].unique())

le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
print(df["Country"].unique())

X = df.drop("Salary", axis=1)
Y = df["Salary"]

# Splitting training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=2)

print(X.shape,X_train.shape,X_test.shape)

# Linear Regression
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
print(linear_reg.fit(X_train, Y_train))

Y_pred_train = linear_reg.predict(X_train)

linear_reg1 = LinearRegression()
print(linear_reg.fit(X_test, Y_test))

Y_pred_test = linear_reg.predict(X_test)

# Accuracy testing on trained and test data
from sklearn.metrics import r2_score
r2_score(Y_train, Y_pred_train)

from sklearn.metrics import mean_squared_error
import numpy as np

error = np.sqrt(mean_squared_error(Y_train, Y_pred_train))
print(error)

from sklearn.metrics import r2_score
r2_score(Y_test, Y_pred_test)

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

dec_tree_reg = DecisionTreeRegressor(random_state=0)
print(dec_tree_reg.fit(X_train, Y_train))

Y_pred_train = dec_tree_reg.predict(X_train)

dec_tree_reg1 = DecisionTreeRegressor(random_state=0)
print(dec_tree_reg.fit(X_test, Y_test))
Y_pred_test = dec_tree_reg.predict(X_test)

# Accuracy test
from sklearn.metrics import r2_score
r2_score(Y_train, Y_pred_train)

error = np.sqrt(mean_squared_error(Y_train, Y_pred_train))
print("${:,.02f}".format(error))

from sklearn.metrics import r2_score
r2_score(Y_test, Y_pred_test)

error = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
print("${:,.02f}".format(error))

# random Forest Regression
from sklearn.ensemble import RandomForestRegressor

random_forest_reg = RandomForestRegressor(random_state=0)
print(random_forest_reg.fit(X_train, Y_train))


Y_pred_train = random_forest_reg.predict(X_train)

random_forest_reg1 = RandomForestRegressor(random_state=0)
print(random_forest_reg.fit(X_test, Y_test))

Y_pred_test = random_forest_reg.predict(X_test)

# Accuracy test
from sklearn.metrics import r2_score
r2_score(Y_train, Y_pred_train)

error = np.sqrt(mean_squared_error(Y_train, Y_pred_train))
print("${:,.02f}".format(error))

from sklearn.metrics import r2_score
r2_score(Y_test, Y_pred_test)

error = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
print("${:,.02f}".format(error))

# Validating with input using Decision Tree
X = np.array([["United States", 'Master’s degree', 15,35]])
print(X)

X[:, 0] = le_country.transform(X[:, 0])
X[:, 1] = le_education.transform(X[:, 1])
X = X.astype(float)
print(X)

y_pred = dec_tree_reg.predict(X)
print(y_pred)


# Extracting the model as pki file using pickle
import pickle

data = {"model": dec_tree_reg, "le_country": le_country, "le_education": le_education}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)

    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)

        regressor_loaded = data["model"]
        le_country = data["le_country"]
        le_education = data["le_education"]

        y_pred = regressor_loaded.predict(X)
        print(y_pred)
