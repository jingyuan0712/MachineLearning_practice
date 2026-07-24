# 📈 Machine Learning Practice: 從線性回歸到隨機森林

歡迎來到本機器學習實作專案！本倉庫（Repository）記錄了我們在機器學習領域中，從最基礎的單變數線性回歸出發，逐步演進至多元線性回歸，並最終升級為非線性隨機森林模型的完整學習歷程與實作細節。

---

## 1. 📌 專案概述 (Project Overview)

本專案是一個**動手實作的機器學習學習計畫**。我們透過解決實際的預測問題（如房價預測），從零開始建構模型，並逐步引入更複雜的特徵與演算法。整個歷程涵蓋了以下三個重要階段：

1. **單變數線性回歸 (Simple Linear Regression)**：利用最簡單的「房屋坪數」單一特徵，預測房屋價格，理解機器學習的最基本預測邏輯。
2. **多元線性回歸 (Multivariate Linear Regression)**：導入包含 20,640 筆資料、8 個特徵的加州房價資料集（California Housing Dataset），處理多維度特徵與特徵縮放。
3. **隨機森林回歸 (Random Forest Regressor)**：突破線性模型的瓶頸，引入樹狀整合模型以捕捉複雜的非線性關係，大幅提升預測精準度。

---

## 2. 🧠 核心機器學習概念 (Core Machine Learning Concepts)

### 🔄 機器學習 vs. 傳統程式設計
在傳統程式設計中，開發者需要手動撰寫「規則」來處理「資料」並產生「答案」。而機器學習則是將「資料」與「答案」輸入給電腦，讓電腦自動尋找並學習出背後的「規則」。

```mermaid
graph TD
    subgraph 傳統程式設計 (Traditional Programming)
        A[資料 Data] --> C[程式/規則 Rules]
        B[規則 Rules] --> C
        C --> D[答案 Answers]
    end

    subgraph 機器學習 (Machine Learning)
        E[資料 Data] --> G[機器學習演算法]
        F[答案 Answers] --> G
        G --> H[規則/模型 Rules]
    end
    
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333,stroke-width:2px
```

### 🎯 機器學習三大核心步驟
每一個監督式學習模型都可以拆解為以下三個核心步驟：

1. **假設模型建立 (Hypothesis Model Formulation)**
   定義特徵與預測值之間的關係模型。例如單變數線性回歸公式：
   $$y = w \cdot x + b$$
   其中 $y$ 為預測目標（標籤），$x$ 為輸入特徵，$w$ 為權重（Weight/斜率），$b$ 為偏差（Bias/截距）。

2. **損失函數 (Loss Function)**
   衡量模型預測值與真實值之間的差距。本專案使用**均方誤差 (Mean Squared Error, MSE)**：
   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
   其中 $y_i$ 為真實值，$\hat{y}_i$ 為預測值，$n$ 為樣本數。我們的目標是讓 $\text{MSE}$ 越小越好。

3. **參數最佳化 (Parameter Optimization)**
   使用**梯度下降法 (Gradient Descent)** 來調整參數 $w$ 與 $b$。透過計算損失函數對參數的偏微分（梯度），沿著梯度相反方向更新參數，直到找到誤差最小的最佳解：
   $$w \leftarrow w - \eta \frac{\partial \text{Loss}}{\partial w}$$
   $$b \leftarrow b - \eta \frac{\partial \text{Loss}}{\partial b}$$
   （其中 $\eta$ 為學習率 Learning Rate）。

---

## 3. 🛠️ 開發環境與安裝設定 (Development Environment & Setup)

### 💻 環境準備
本專案使用 Python 虛擬環境進行隔離，以確保套件版本的穩定與乾淨。

1. **建立虛擬環境**：
   ```powershell
   python -m venv .venv
   ```
2. **啟用虛擬環境**：
   - **Windows PowerShell**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```

> [!IMPORTANT]
> **Windows PowerShell 執行原則修正：**
> 若在 Windows 上啟用虛擬環境時遇到「因為此系統上已停用指令碼執行，所以無法載入...」的錯誤，請以系統管理員身分或在當前視窗中執行以下指令來暫時解除限制：
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```

3. **安裝必要套件**：
   ```bash
   pip install scikit-learn numpy pandas
   ```

---

## 4. 📈 實作演進歷程 (Practical Implementation & Evolution)

### 📍 階段一：單變數線性回歸
* **實作程式**：[main.py](file:///c:/Users/program%20file/MachineLearning_practice/main.py)
* **應用情境**：根據房屋的「坪數」來預測「房價」（萬元）。
* **核心程式碼**：
  ```python
  import numpy as np
  from sklearn.linear_model import LinearRegression

  # 準備數據 (坪數 vs 房價)
  x = np.array([[10], [15], [20], [25], [30], [35], [40]])
  y = np.array([300, 420, 500, 600, 710, 800, 920])

  # 訓練模型
  model = LinearRegression()
  model.fit(x, y)
  ```
* **訓練結果**：
  * 電腦學習出來的公式為：
    $$y \approx 20.21x + 101.79$$
    這代表房屋的基本底價約為 $101.79$ 萬元，而每增加一坪，房價平均上漲 $20.21$ 萬元。

---

### 📍 階段二：多元線性回歸
* **實作程式**：[Linear Regression.py](file:///c:/Users/program%20file/MachineLearning_practice/Linear%20Regression.py)
* **資料集說明**：使用加州房價資料集（California Housing Dataset），包含 20,640 筆資料，共有 8 個特徵（如中位數收入、房齡、房間數、地理座標等）。
* **關鍵技術**：
  * **資料集拆分**：使用 `train_test_split` 將資料以 80/20 比例拆分為訓練集與測試集，確保評估的客觀性。
  * **特徵標準化 (`StandardScaler`)**：由於各特徵尺度差異極大（如房齡只有幾十年，人口卻有數千人），標準化能加快梯度下降收斂速度並避免大尺度特徵主導模型。
* **核心程式碼**：
  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LinearRegression

  # 拆分資料集
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # 特徵縮放
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # 訓練模型
  model = LinearRegression()
  model.fit(X_train_scaled, y_train)
  ```
* **模型表現**：
  * **決定係數 $R^2$**：$\approx 0.5758$ (代表模型能解釋約 57.58% 的房價變異)
  * **預測平均誤差 (RMSE)**：$\approx \$74,558$ 美元
* **線性模型的瓶頸反思**：
  真實世界中的房價與特徵關係極其複雜且通常是**非線性**的（例如：經緯度與房價的關係呈非線性，高緯度某些特定區域特別貴，而非單純的緯度越高房價越貴）。線性模型強迫使用平面（超平面）去擬合數據，會遇到嚴重的**欠擬合 (Underfitting)**，這也是 $R^2$ 分數停留在 0.57 左右的瓶頸所在。

---

### 📍 階段三：非線性隨機森林回歸 (強勢升級)
* **實作程式**：[Random Forest.py](file:///c:/Users/program%20file/MachineLearning_practice/Random%20Forest.py)
* **核心模型**：`RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`，由 100 棵決策樹組成的整合學習模型，能自動發掘並建構特徵之間的複雜非線性組合關係。
* **核心程式碼**：
  ```python
  from sklearn.ensemble import RandomForestRegressor

  # 初始化並訓練隨機森林模型
  model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
  model.fit(X_train_scaled, y_train)
  ```
* **模型表現**：
  * **決定係數 $R^2$**：$\approx 0.8047$ (相較線性回歸顯著提升！)
  * **預測平均誤差 (RMSE)**：$\approx \$50,588$ 美元 (誤差大幅降低！)
* **🔑 關鍵機制差異：權重 vs. 特徵重要性**：
  * **線性模型權重 (`coef_`)**：表示特徵與目標變數之間的**方向性**與**線性比例關係**。正值代表正相關，負值代表負相關。
  * **樹狀模型特徵重要性 (`feature_importances_`)**：表示該特徵在所有決策樹中，對於「減少預測誤差（降不純度/減少方差）」的**貢獻度總和**。數值永遠為正數（範圍為 0~1 之間，總和為 100%），無法直接看出是正向還負向影響，但能精確指出「哪些變數最具備判別力」。

---

## 5. 📊 模型效能比較 (Model Performance Comparison)

| 評估指標 / 模型 | 階段一：單變數線性回歸 | 階段二：多元線性回歸 | 階段三：隨機森林回歸 |
| :--- | :--- | :--- | :--- |
| **輸入特徵數 (Features)** | 單一特徵 (坪數) | 8 個特徵 | 8 個特徵 |
| **決定係數 ($R^2$ Score)** | 基準點 (Baseline) | $\approx 0.5758$ | **$\approx 0.8047$ (🚀 效能提升約 +22.9%)** |
| **平均預測誤差 (RMSE)** | N/A | $\approx \$74,558$ 美元 | **$\approx \$50,588$ 美元 (📉 誤差減少約 -$24,000$)** |
| **特徵分析工具** | 斜率 ($w$) | 線性係數 (`coef_`) | 特徵重要性分數 (`feature_importances_`) |

---

## 6. 🔑 核心啟示與特徵洞察 (Key Takeaways)

透過觀察隨機森林模型的 `feature_importances_`，我們發現：
* **地理座標 (`Latitude` / `Longitude`)** 與 **中位數收入 (`MedInc`)** 合計貢獻了超過 **70%** 的特徵重要性！
* 這高度呼應並印證了房地產界的名言：**「地段、地段、還是地段！ (Location, Location, Location!)」**
* 多元線性回歸因無法有效處理「經緯度」這類地理空間的複雜非線性關係，導致預測精準度受限；而隨機森林回歸能自動在不同區域進行樹狀切割，完美捕捉了位置所帶來的非線性高溢價與低溢價區，這也是模型效能大幅躍進的最主要關鍵。

---
