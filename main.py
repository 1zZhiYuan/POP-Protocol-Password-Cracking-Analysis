import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_recall_curve, roc_curve, auc,
    average_precision_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

# 1. 加载数据
train_data = pd.read_csv("kd99.csv")

# 2. 数据采样（可选，用于加速调试）
# 如果你不需要加速调试，可以删除这一行
# train_data = train_data.sample(frac=0.5, random_state=42)  # 使用 50% 的训练数据

# 3. 数据预处理
categorical_features = ['protocol_type', 'service', 'flag']
label_encoder = LabelEncoder()

# 对分类特征进行编码
for col in categorical_features:
    train_data[col] = label_encoder.fit_transform(train_data[col])

# 对目标列编码
label_encoder = LabelEncoder()
train_data['type'] = label_encoder.fit_transform(train_data['type'])

# 4. 拆分训练集和测试集（70% 训练集，30% 测试集）
X = train_data.drop(columns=['type'])
y = train_data['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 二分类：将 "normal" 设置为正类，其他为负类
target_class = label_encoder.transform(['normal'])[0]
y_train_binary = (y_train == target_class).astype(int)
y_test_binary = (y_test == target_class).astype(int)

# 5. 手动设置超参数
clf = DecisionTreeClassifier(
    criterion='entropy',         # 分裂标准
    max_depth=12,             # 最大深度
    min_samples_split=4,      # 最小样本分裂数
    min_samples_leaf=4,       # 最小叶子节点数
    random_state=42,
    max_features=None,        # 自动选择特征数
    max_leaf_nodes=None       # 自动选择最大叶节点数
)

# 6. k-折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_score_mean = cross_val_score(clf, X_train, y_train_binary, cv=cv, scoring='accuracy').mean()
print(f"Cross-Validation Accuracy: {cross_val_score_mean:.4f}")

# 7. 模型训练
clf.fit(X_train, y_train_binary)

# 8. 模型预测
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# 9. 绘制 ROC 曲线
fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label="Random Guess")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve (Best Decision Tree)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# 10. 绘制 PR 曲线
precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Best Decision Tree)")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.show()

# 11. 性能评估
accuracy = accuracy_score(y_test_binary, y_pred)
recall = recall_score(y_test_binary, y_pred)
f1 = f1_score(y_test_binary, y_pred)  # 计算 F1 值
# 误报率 (False Positive Rate)
fpr_final = fpr[1]  # FPR is typically calculated from the ROC curve

# 显示性能
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"False Positive Rate: {fpr_final:.4f}")

# 12. 计算 PR 曲线的平均精度
average_precision = average_precision_score(y_test_binary, y_pred_proba)
print(f"Average Precision: {average_precision:.4f}")