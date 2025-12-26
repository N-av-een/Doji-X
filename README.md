# Doji-X: Algorithmic Trading System with Machine Learning 📈

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

**Doji-X** is a "Glass Box" algorithmic trading decision support system designed for the Gold Futures (GC=F) market. It hybridizes **Heikin Ashi** noise reduction with a **Random Forest** Machine Learning classifier to identify high-probability trade setups and filter out false signals (Bull/Bear traps).



---

## 🚀 Key Features

* **Glass Box Interface:** Unlike "Black Box" systems, Doji-X visualizes the *Confidence Score* and *Feature Importance* for every signal.
* **Heikin Ashi Smoothing:** Automatically filters market noise to reveal the true trend.
* **ML-Powered Filtering:** Uses a Random Forest Classifier to reject "False Breakouts" that traditional technical analysis would miss.
* **Real-Time Dashboard:** Built with Streamlit for interactive charting and live probability analysis.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Data Processing:** Pandas, NumPy
* **Technical Indicators:** TA-Lib / Pandas-TA
* **Data Source:** Yahoo Finance API (`yfinance`)

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally on your machine.
