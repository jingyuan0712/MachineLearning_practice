import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. 載入大型資料集 (California Housing)
# ==========================================
print("📦 正在載入 California Housing 大型資料集 (包含 20,640 筆資料)...")
housing = fetch_california_housing()

X = housing.data           # 8 個輸入特徵矩陣
y = housing.target         # 房價 (單位: 10萬美元)
feature_names = housing.feature_names  # 特徵名稱列表

print("\n📊 特徵名稱與說明：")
print("- MedInc: 地區中位數收入 | HouseAge: 房屋平均屋齡 | AveRooms: 平均房間數")
print("- AveBedrms: 平均臥室數 | Population: 人口數 | AveOccup: 平均每戶人數")
print("- Latitude / Longitude: 緯度 / 經度")

# ==========================================
# 2. 特徵與標籤分離 + 資料集拆分 (Train/Test Split)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✂️ 資料拆分完成：訓練集 {X_train.shape[0]} 筆，測試集 {X_test.shape[0]} 筆")

# ==========================================
# 3. 特徵標準化 (Feature Scaling)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. 建立與訓練多元線性回歸模型
# ==========================================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ==========================================
# 5. 模型評估 (Evaluation)
# ==========================================
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n🎉 模型訓練與評估完成！")
print("--------------------------------------------------")
print(f"📈 模型 R² Score (準確度分數): {r2:.4f} (越接近 1 代表預測力越好)")
print(f"📉 預測平均誤差 (RMSE): ${rmse * 100000:,.2f} 美元")
print("--------------------------------------------------")

# ==========================================
# 6. 觀察各個特徵對房價的影響力 (Feature Importance)
# ==========================================
print("\n🔍 各個變數對房價的影響力 (權重 Coefficients)：")
print(f"{'特徵名稱 (Feature)':<20} | {'權重 (Weight)':<15}")
print("-" * 40)

# 將特徵與權重配對排序
importance = sorted(zip(feature_names, model.coef_), key=lambda x: abs(x[1]), reverse=True)
for name, coef in importance:
    print(f"{name:<20} | {coef:+.4f}")

# ==========================================
# 7. 進行單筆資料實測
# ==========================================
print("\n🏠 隨機抽樣 1 筆測試集資料進行真實預測比對：")
sample_index = 0
sample_x = X_test_scaled[sample_index].reshape(1, -1)
actual_price = y_test[sample_index] * 100000
predicted_price = model.predict(sample_x)[0] * 100000

print(f"真實房價: ${actual_price:,.2f} 美元")
print(f"模型預測: ${predicted_price:,.2f} 美元")
print(f"相差金額: ${abs(actual_price - predicted_price):,.2f} 美元")