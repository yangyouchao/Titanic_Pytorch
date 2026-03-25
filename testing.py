import pandas as pd
import numpy as np
import training
from pandas.core.frame import DataFrame

data_test = pd.read_csv(R"C:\Users\huluye\Desktop\titanic_pytorch\processed_data\df_test.csv")
y1 = training.onehot_enc(data_test, 'Sex')
y2 = training.onehot_enc(data_test, 'Embarked')
remain_data = data_test[['Pclass', 'Age']]
x_test = np.hstack((remain_data, y1, y2))          
x_test_ts = training.torch.from_numpy(x_test)      
x_test_ts = x_test_ts.float()
print(x_test_ts[0:3])
model = training.ClassificationModel(x_test.shape[1], num_classes=1)
model.load_state_dict(training.torch.load('model.pth'))
model.eval()
with training.torch.no_grad():
    output = model(x_test_ts)

survived = []
length = len(output)
for i in range(length):
    if output[i] > 0.5:
        survived.append(1)
    else:
        survived.append(0)
Survived = DataFrame(survived)
Survived.columns = ['Survived']
PassengerId = data_test['PassengerId']
Result = pd.concat([PassengerId, Survived], axis=1)
Result.to_csv(R"C:\Users\huluye\Desktop\titanic_pytorch\processed_data\result.csv", index=False)

#Survived.to_csv(R"C:\Users\huluye\Desktop\titanic_pytorch\processed_data\result.csv", mode='a',header='Survived', index=False)
# col_name = .columns.tolist()
# valid_cols = [c for c in col_name if c != 'Unnamed: 0']
# data_test = pd.read_csv(R"C:\Users\huluye\Desktop\titanic_pytorch\processed_data\df_test.csv", usecols=valid_cols)
# if data_test.empty:
#     print('导入文件是空的')
# str_columns = data_test.select_dtypes(include=['str']).columns.tolist()r

# for col in data_test.columns.tolist():
#     if col in str_columns:
#         print('需要独热编码')
#     else:
#         print('不需要')
# for col in data_test.columns.tolist():
#     print(col)

#model_test = training.ClassificationModel()

# sex_onehot = training.onehot_enc(data_test, col='Sex')
# print(type(sex_onehot))
