import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 設定隨機種子，確保每次執行的實驗結果一致
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 載入並準備資料集
# ==========================================
print("📦 正在載入 California Housing 大型資料集...")
housing = fetch_california_housing()

X = housing.data
y = housing.target.reshape(-1, 1)  # 轉為二維陣列 [N, 1] 以符合 PyTorch 輸出格式

# 資料集拆分 (80% 訓練集, 20% 測試集)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特徵標準化 (Standardization) - 神經網路對數值範圍非常敏感！
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 💡 關鍵步驟：將 NumPy 陣列轉換為 PyTorch Tensor (張量)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print(f"✂️ 資料轉換完成！訓練集張量維度: {X_train_tensor.shape}")

# ==========================================
# 2. 定義神經網路結構 (Build Neural Network)
# ==========================================
class HousingMLP(nn.Module):
    def __init__(self, input_dim):
        super(HousingMLP, self).__init__()
        # 輸入層 (8 個特徵) -> 第一隱藏層 (64 個神經元)
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()  # 激活函數：注入非線性轉折
        
        # 第一隱藏層 (64) -> 第二隱藏層 (32 個神經元)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        
        # 第二隱藏層 (32) -> 輸出層 (1 個神經元，代表預測房價)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # 前向傳播 (Forward Pass)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型
input_features = X_train_tensor.shape[1]  # 8 個特徵
model = HousingMLP(input_dim=input_features)

print("\n🧠 神經網路結構已建立：")
print(model)

# ==========================================
# 3. 設定損失函數與優化器
# ==========================================
criterion = nn.MSELoss()  # 均方誤差損失函數
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam 優化器 (學習率 0.01)

# ==========================================
# 4. 訓練模型 (Training Loop)
# ==========================================
epochs = 200  # 全套資料訓練 200 輪
print("\n🚀 開始訓練神經網路 (Training)...")

for epoch in range(1, epochs + 1):
    model.train()
    
    # 1. 前向傳播 (Forward)
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    
    # 2. 反向傳播 (Backward Pass)
    optimizer.zero_grad()  # 清空前一次的梯度紀錄
    loss.backward()        # 計算梯度
    optimizer.step()       # 根據梯度更新神經元權重
    
    # 每 20 輪印出一次損失降低情況
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch [{epoch:>3d}/{epochs}] | Loss (MSE): {loss.item():.4f}")

# ==========================================
# 5. 模型評估與推論 (Evaluation)
# ==========================================
model.eval()  # 切換為評估模式
with torch.no_grad():  # 關閉梯度計算以節省記憶體
    test_predictions = model(X_test_tensor).numpy()

y_test_np = y_test_tensor.numpy()

# 計算評估指標
mse = mean_squared_error(y_test_np, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_np, test_predictions)

print("\n🎉 PyTorch 神經網路訓練與評估完成！")
print("--------------------------------------------------")
print(f"📈 模型 R² Score (準確度分數): {r2:.4f}")
print(f"📉 預測平均誤差 (RMSE): ${rmse * 100000:,.2f} 美元")
print("--------------------------------------------------")

# 隨機抽樣比對
sample_index = 0
actual_price = y_test_np[sample_index][0] * 100000
predicted_price = test_predictions[sample_index][0] * 100000
print(f"🏠 抽樣比對 - 真實房價: ${actual_price:,.2f} 美元")
print(f"🏠 抽樣比對 - 模型預測: ${predicted_price:,.2f} 美元")
print(f"相差金額: ${abs(actual_price - predicted_price):,.2f} 美元")