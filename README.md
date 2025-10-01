# HW1-1: 互動式線性迴歸視覺化工具 (Interactive Linear Regression Visualizer)

## 專案簡介

這是一個使用 **Streamlit** 開發的互動式 Web 應用程式，用於視覺化線性迴歸的過程。使用者可以調整數據生成參數（如數據點數量、係數和雜訊量），即時觀察數據分佈、模型擬合的迴歸線，以及自動識別出的主要離群值。

## 專案特色

1.  **數據生成可控性 (Data Generation):**
    * 使用者可選擇數據點數量 $n$ (100 - 1000)。
    * 可調整真實係數 $a$ (-10 到 10)。
    * 可調整雜訊變異數 $var$ (0 到 1000)，雜訊服從 $N(0, var)$ 分佈。
    * 數據關係：$y = ax + b + noise$ ($b$ 固定為 5)。
2.  **線性迴歸視覺化 (Visualization):**
    * 繪製所有生成的數據點。
    * 以**紅色**繪製由 `scikit-learn` 計算出的最佳擬合迴歸線。
3.  **離群值識別 (Outlier Identification):**
    * 自動計算殘差 (Residuals)。
    * 識別並標記 **Top 5** 離群值（殘差絕對值最大的 5 個點）。
4.  **互動式介面 (User Interface):**
    * 所有參數皆透過 Streamlit 側邊欄的滑動條進行調整。
    * 即時顯示迴歸結果（擬合係數 $\hat{a}$ 和 $R^2$ Score）。

## 如何執行 (Local Setup)

### 1. 建立虛擬環境 (Virtual Environment)

```bash
python -m venv venv
# 啟用環境 (Windows)
.\venv\Scripts\activate
# 啟用環境 (macOS/Linux)
source venv/bin/activate