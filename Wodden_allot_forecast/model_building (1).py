import numpy as np
import pandas as pd

data = pd.read_excel('wodden_model.xlsx')
# Preprocess the data, if necessary

data.head()

data.dtypes

# Convert the datetime column to datetime type
data["Forecasted Date"] = pd.to_datetime(data["Forecasted Date"])

# Extract the year, month, day
data["year"] = data["Forecasted Date"].dt.year

data["month"] = data["Forecasted Date"].dt.month
data["day"] = data["Forecasted Date"].dt.day

data.columns

# Drop the original datetime column
data = data.drop("Forecasted Date", axis=1)

# Label Encoder
from sklearn.preprocessing import LabelEncoder

# Creating instance of labelencoder
labelencoder = LabelEncoder()

data['Customer/Vendor Name'] = labelencoder.fit_transform(data['Customer/Vendor Name'])

data.shape

data.rename(columns={'Customer/Vendor Name': 'customername'}, inplace=True)

data.columns

data = data.reindex(columns=[*data.columns[:5], 'predicted', *data.columns[4:]])


#data["year"], year_labels = data["year"].factorize()
#data["month"], month_labels = data["month"].factorize()
#data["day"], day_labels = data["day"].factorize()

#data.head()

# Importing the libraries for model building
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

data.columns

X = data.iloc[:, :4]

y = data.iloc[:, 4]


from sklearn.linear_model import LinearRegression
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(data, test_size = 0.2) # 20% test data

regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

data.columns

# preparing the model on train data 
model_train = smf.ols("predicted ~ customername + year + month + day", data = data_train).fit()
model_train.summary()

# prediction on test data set 
test_pred = model_train.predict(data_test)

# test residual values 
test_resid = test_pred - data_test.predicted

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(data_train)

# train residual values
train_resid  = train_pred - data_train.predicted

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

# Saving model to disk
pickle.dump(regressor, open('model1.pkl','wb'))

# Loading model to compare the results
model1 = pickle.load(open('model1.pkl','rb'))

print(model1.predict([[0, 2022, 2, 12]]))