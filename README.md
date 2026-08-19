# 📉 Customer Churn Analysis Dashboard

Predicting which telecom customers are likely to churn using the IBM Telco dataset.  
Combines **exploratory data analysis**, **machine learning**, and an **interactive Streamlit dashboard**.

---

## 🔍 Key Findings

| Insight | Detail |
|---|---|
| Overall churn rate | **26.5%** (1,869 / 7,043 customers) |
| Biggest driver | Month-to-month contracts → **42% churn rate** |
| Highest-risk window | Customers in first 12 months → **47% churn** |
| Fiber optic problem | **41% churn** vs 19% for DSL |
| Revenue at risk | **~$139K/month** in lost MRR |

---

## 📊 Dashboard Preview

> Upload the Kaggle CSV and get instant EDA + ML results in your browser.

The dashboard includes:
- KPI cards (churn rate, MRR at risk, avg tenure)
- Exploratory charts — contract type, tenure, monthly charges, internet service
- Confusion matrix + ROC curve
- Top 15 feature importances
- Actionable business recommendations

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-analysis.git
cd customer-churn-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Get the CSV from Kaggle:  
👉 [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Place it at: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 4. Run the Streamlit dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Or run the notebook
```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

---

## 🗂️ Project Structure

```
customer-churn-analysis/
├── data/                        ← Add your CSV here (gitignored)
├── notebooks/
│   └── churn_analysis.ipynb     ← Full EDA + model walkthrough
├── src/
│   └── churn_model.py           ← Preprocessing, training, evaluation
├── dashboard/
│   └── app.py                   ← Streamlit app
├── assets/                      ← Saved plots from the notebook
├── model/                       ← Saved model (gitignored)
├── requirements.txt
└── README.md
```

---

## 🤖 Model Details

| | |
|---|---|
| Algorithm | Random Forest (200 trees) |
| Train / test split | 80 / 20 (stratified) |
| Class imbalance | Handled with `class_weight='balanced'` |
| Accuracy | 79.7% |
| AUC–ROC | **84.3%** |
| Precision | 73.3% |
| Recall | 53.3% |
| F1 Score | 61.8% |

---

## 🛠 Tech Stack

- **Python 3.10**
- pandas, numpy — data manipulation
- scikit-learn — ML model
- matplotlib, seaborn — visualizations
- Streamlit — interactive dashboard
- Jupyter — exploratory notebook

---

## 📦 Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo → `dashboard/app.py`
4. Deploy — you'll get a public URL to share

---

## 📄 License

MIT License — free to use, modify, and distribute.

