import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np


# 定义DNN模型
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.rnn(x.unsqueeze(1))  # 如果x是二维的，添加一个大小为1的维度来模拟序列长度
        x = self.fc(x[:, -1, :])  # 然后取序列的最后一个时间步
        return x


# 确保文件路径正确，并且所有文件都可访问
file_paths = [
    "F:\\LLL\\MachineLearningCVE\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "F:\\LLL\\MachineLearningCVE\\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "F:\\LLL\\MachineLearningCVE\\Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "F:\\LLL\\MachineLearningCVE\\Monday-WorkingHours.pcap_ISCX.csv",
    "F:\\LLL\\MachineLearningCVE\\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "F:\\LLL\\MachineLearningCVE\\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "F:\\LLL\\MachineLearningCVE\\Tuesday-WorkingHours.pcap_ISCX.csv",
    "F:\\LLL\\MachineLearningCVE\\Wednesday-workingHours.pcap_ISCX.csv"
]

# 读取所有CSV文件并合并到一个DataFrame中
df_list = [pd.read_csv(file_path) for file_path in file_paths]
df = pd.concat(df_list)

# 使用replace方法替换'label'列中的值,主要把数据集进行分类大类
df[' Label'] = df[' Label'].replace(['Web Attack � Brute Force', 'Web Attack � XSS', 'Web Attack � Sql Injection'], 'web attack')
df[' Label'] = df[' Label'].replace(['DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye', 'Heartbleed'], 'DoS')
df[' Label'] = df[' Label'].replace(['FTP-Patator', 'SSH-Patator'], 'Brute Force')
df[' Label'] = df[' Label'].replace(['Infiltration'], 'PortScan')
print('=======================================查看数据的label分布情况==========================')
print(df[' Label'].unique())
print(df[' Label'].value_counts())
num_features = df.shape[1]
print(f"数据集的特征列数量为: {num_features}")
print(df.head())

# 替换label列中的值并初始化LabelEncoder对象并进行标签编码
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df[' Label'])

# 打印编码后的DataFrame
print(df)
print('查看数据集标签特征')
print(df['label_encoded'].value_counts())

# 删除'label'列
df = df.drop(columns=[' Label'])
print(df)

# 生成新的列名列表，从1到79
new_column_names = list(range(1, 80))

# 检查新列名的数量是否与原始列数匹配
if len(new_column_names) != df.shape[1]:
    raise ValueError("新列名数量与原始列数不匹配")

# 将DataFrame的列名替换为新列名
df.columns = new_column_names

# 可以将修改后的DataFrame保存到新的CSV文件
df.to_csv('IDS2017-1.csv', index=False)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# 假设最后一列是标签编码后的数值
X = df.iloc[:, :-1].astype(float)  # 确保所有特征列都是浮点数
y = df.iloc[:, -1].astype(int)  # 标签列已经是整数类型
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
'================================================进行数据归一化============================================='
scaler = StandardScaler()
cols = x_train.select_dtypes(include=['float64', 'int64']).columns
sc_train = scaler.fit_transform(x_train.select_dtypes(include=['float64', 'int64']))
sc_test = scaler.fit_transform(x_test.select_dtypes(include=['float64', 'int64']))
sc_traindf = pd.DataFrame(sc_train, columns=cols)
sc_testdf = pd.DataFrame(sc_test, columns=cols)
print('===================================================查看数据归一化之后的数据集====================================')
print(sc_traindf)
print(sc_testdf)
x_train = sc_traindf
x_test = sc_testdf

# 将数据转换为PyTorch Tensor
X_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# 创建TensorDataset和DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 实例化模型
input_size = X_train_tensor.size(1)  # 特征数量
hidden_size = 128  # 隐藏层大小
num_classes = len(np.unique(y_train))  # 类别数量

dnn_model = DNN(input_size, hidden_size, num_classes)
rnn_model = RNN(input_size, hidden_size, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer_dnn = optim.Adam(dnn_model.parameters(), lr=0.001)
optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=0.001)


# 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs} Loss: {total_loss / len(train_loader):.4f}')


# 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

'=================================随机森林算法=========================================='
x_train_np = x_train.values.astype(np.float32)
x_test_np = x_test.values.astype(np.float32)

# 训练和评估随机森林模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 对于分类问题使用训练数据拟合模型
clf.fit(x_train_np, y_train)

# 对于分类问题对测试集进行预测
y_pred_rf = clf.predict(x_test_np)

# 评估模型性能
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
rf_confusion_matrix = confusion_matrix(y_test, y_pred_rf)
rf_classification_report = classification_report(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:\n", rf_confusion_matrix)
print("Random Forest Classification Report:\n", rf_classification_report)

# 训练和评估DNN模型
train_model(dnn_model, train_loader, criterion, optimizer_dnn, epochs=5)
dnn_accuracy = evaluate_model(dnn_model, test_loader)
print(f"DNN Model Accuracy: {dnn_accuracy:.2f}%")

# 训练和评估RNN模型
train_model(rnn_model, train_loader, criterion, optimizer_rnn, epochs=5)
rnn_accuracy = evaluate_model(rnn_model, test_loader)
print(f"RNN Model Accuracy: {rnn_accuracy:.2f}%")
