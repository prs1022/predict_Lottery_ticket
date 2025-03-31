from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 假设你有一些历史的双色球数据，每组数据包含红球号码和蓝球号码
# 这里的数据是随机生成的示例数据
np.random.seed(42)
data_size = 1000
red_balls = np.random.choice(range(1, 34), size=(data_size, 6), replace=False)
blue_balls = np.random.choice(range(1, 17), size=(data_size, 1), replace=False)

# 将数据组合成特征矩阵和标签
features = np.concatenate([red_balls, blue_balls], axis=1)
labels = np.random.randint(2, size=data_size)  # 随机生成标签，这里仅作示例

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 使用随机森林分类器作为模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 进行预测
predictions = model.predict(X_test)

# 输出准确率
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
