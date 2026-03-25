import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torchvision.models as models

#定义模型
class ClassificationModel(nn.Module):
    def __init__(self, input_features:int, num_classes:int):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_features, 32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x 

#独热编码
def onehot_enc(pd_dataframe:pd.DataFrame, col:str):
    ori = pd_dataframe[col].values
    onehot = pd.get_dummies(ori, columns=[col], prefix=[col])
    features = onehot.columns.tolist()
    nump = onehot[features].values.astype('double')
    return nump

#模型训练
def model_train(NPArray, Train_data, Target_data, Learning_rate:float, Epochs:int):
    #初始化模型
    model = ClassificationModel(input_features=NPArray.shape[1], num_classes=1)

    #转移到GPU上进行训练
    #print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    x_train_ts = Train_data.to(device)
    y_train_ts = Target_data.to(device)

    #构建损失函数和优化器
    critertion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=Learning_rate)

    for epoch in range(Epochs):
        outputs = model(x_train_ts)
        if epoch == 9999:
            print('outputs',outputs)
        loss = critertion(outputs, y_train_ts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(epoch+1)%1000 == 0:
            print(f'Epoch: [{epoch+1}], Loss: {loss.item():.4f}')
    #保存模型,仅保存起参数
    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    data_train = pd.read_csv(R'C:\Users\huluye\Desktop\titanic_pytorch\processed_data\df_train.csv')
    
    #独热编码str类型
    x1 = onehot_enc(data_train, col='Sex')
    x2 = onehot_enc(data_train, col='Embarked')

    #导入剩下的年龄和乘客等级，构成完整的训练数据集
    remain_data = data_train[['Pclass', 'Age']]
    x_train = np.hstack((remain_data, x1, x2))         
    x_train_ts = torch.from_numpy(x_train)      
    x_train_ts = x_train_ts.float()           
    y_train_ts = torch.tensor(data_train['Survived'])   
    y_train_ts = y_train_ts.unsqueeze(1)                #将y_train_ts从[889]调整为[889, 1]
    y_train_ts = y_train_ts.float()

    model_train(x_train, Train_data=x_train_ts, Target_data=y_train_ts, Learning_rate=0.004, Epochs=10000)
