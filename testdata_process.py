import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

data_test = pd.read_csv(R"C:\Users\huluye\Desktop\titanic_pytorch\data\test.csv")
print(data_test.info())

df = data_test[['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
train = df[df['Age'].notna()]     #提取出Age没缺失的样本，用于训练
predict = df[df['Age'].isna()]    #提取出Age缺失的样本

x_train = train.drop('Age', axis=1)
y_train = train['Age']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

x_predict = predict.drop('Age', axis=1)
data_predAge = rf.predict(x_predict)

data_test.loc[data_test['Age'].isna(), 'Age'] = data_predAge

df_test = data_test.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'])
df_test.to_csv(R"C:\Users\huluye\Desktop\titanic_pytorch\processed_data\df_test.csv", index=False)
