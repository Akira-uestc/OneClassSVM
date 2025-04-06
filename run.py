import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from model import OneClassSVM


def load_data():
    train_data = pd.read_csv("./Train_data.csv")
    test_data1 = pd.read_csv("./Test_data2.csv")
    test_data2 = pd.read_csv("./Test_data1.csv")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test1_scaled = scaler.transform(test_data1)
    test2_scaled = scaler.transform(test_data2)

    return train_scaled, test1_scaled, test2_scaled, test_data1.index, test_data2.index


def visualize_scores(scores, threshold, title, index=None):
    plt.figure(figsize=(10, 5))
    if index is None:
        index = range(len(scores))
    scores = scores.flatten()
    plt.plot(index, scores, marker="o", linestyle="-", label="Decision Scores")

    # 控制限线
    plt.axhline(y=threshold, color="r", linestyle="--", label="Control Limit (0)")

    # 标出异常点
    outliers = [i for i, s in zip(index, scores) if s < threshold]
    plt.scatter(outliers, [scores[i] for i in outliers], color="red", label="Anomalies")

    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Decision Function Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    train_data, test_data1, test_data2, idx1, idx2 = load_data()

    # 初始化 & 拟合模型
    model = OneClassSVM(nu=0.1, gamma=0.1)
    model.fit(train_data)

    print("Model fitted.")

    # 计算测试集得分
    scores1 = model.decision_function(test_data1)
    print(f"Test Data 1 Decision Function Scores: {scores1.flatten()}")
    scores2 = model.decision_function(test_data2)
    print(f"Test Data 2 Decision Function Scores: {scores2.flatten()}")

    # 控制限为0（decision_function >= 0 表示正常）
    threshold = 0.0

    # 画图展示
    visualize_scores(scores1, threshold, "Test Data 1 - Anomaly Detection", idx1)
    visualize_scores(scores2, threshold, "Test Data 2 - Anomaly Detection", idx2)


if __name__ == "__main__":
    main()
