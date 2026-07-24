import numpy as np
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. 準備資料集 (Dataset)
# ==========================================
# x: 房屋坪數 (特徵 Feature) - 必須是二維陣列 (Shape: [樣本數, 特徵數])
x = np.array([[10], [15], [20], [25], [30], [35], [40]])

# y: 對應房價 (標籤 Label/Target) - 單位：萬元
y = np.array([300, 420, 500, 600, 710, 800, 920])

# ==========================================
# 2. 建立並訓練模型 (Model Training)
# ==========================================
# 初始化線性回歸模型 (定義 Hypothesis: y = w*x + b)
model = LinearRegression()

# 開始訓練！(電腦在這裡自動執行「計算誤差 + 梯度下降」，尋找最佳的 w 與 b)
model.fit(x, y)

# ==========================================
# 3. 檢視電腦學出來的「規則」
# ==========================================
w = model.coef_[0]        # 權重 (Weight/斜率)
b = model.intercept_      # 偏差 (Bias/截距)

print("🎉 模型訓練完成！")
print(f"電腦學習出來的公式為: 房價 = {w:.2f} * 坪數 + {b:.2f}")
print("--------------------------------------------------")

# ==========================================
# 4. 進行新資料預測 (Inference / Prediction)
# ==========================================
# 假設我們想預測一間「28 坪」和「50 坪」的房子大概值多少錢
new_houses = np.array([[28], [50]])
predicted_prices = model.predict(new_houses)

for house_size, price in zip(new_houses, predicted_prices):
    print(f"🏠 預測坪數 {house_size[0]} 坪的房子價格為: {price:.1f} 萬元")