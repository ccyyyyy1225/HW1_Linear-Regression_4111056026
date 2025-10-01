# 🚀 部署指南: 部署至 Replit

本指南詳細說明將 Streamlit 應用程式 (`app.py`) 部署到 Replit 平台所需的步驟。

## 前提條件

1.  擁有一個 Replit 帳戶。
2.  所有專案檔案 (`app.py`, `requirements.txt`) 已推送至 GitHub 儲存庫，或直接上傳至 Replit 專案中。

## Replit 部署步驟

### 1. 建立新專案

* 前往 Replit 並建立一個**新 Repl**。
* 選擇 **Python** 模板。
* **可選（推薦）：** 如果您的程式碼在 GitHub 上，請選擇**從 GitHub 導入 (Import from GitHub)** 並貼上儲存庫 URL。

### 2. 確認相依套件 (`requirements.txt`)

Replit 會自動處理 `requirements.txt` 中的套件安裝。

* **請確保 `requirements.txt` 內容正確：**
    ```
    streamlit
    numpy
    pandas
    scikit-learn
    matplotlib
    ```

### 3. 設定執行指令

Replit 需要知道如何啟動 Streamlit 伺服器。

1.  找到 **`.replit`** 配置文件（可能需要查看隱藏檔案）。
2.  將 `run` 指令修改為使用 Streamlit 執行您的主檔案：

    ```
    run = "streamlit run app.py"
    ```

### 4. 運行與發布

1.  點擊 **Run (運行)** 按鈕。
2.  Replit 將會安裝套件並執行 Streamlit 指令。
3.  Streamlit 應用程式將載入到嵌入式的網頁檢視窗格中。**此窗格中顯示的 URL 即為您的即時部署 URL。**
4.  確認應用程式運行正常後，即可使用此 URL 分享您的專案。