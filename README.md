# 📊 Customer Churn Prediction & Analytics Engine

An end-to-end Machine Learning web application designed to analyze customer behavior and predict the probability of churn for telecommunications / subscription-based businesses. Built with **Python**, **Scikit-Learn**, and **Streamlit**, this tool helps business teams identify high-risk customers early and implement data-driven retention strategies.

---

## 🌟 Key Features

* **Real-time Churn Prediction:** Input individual customer metrics to receive instant churn risk probabilities (High, Medium, or Low Risk).
* **Interactive Dashboard:** Built with Streamlit for a seamless, intuitive user experience.
* **Feature Importance Insights:** Visualizes key factors driving customer churn (e.g., tenure, contract type, monthly charges, support tickets).
* **Automated Data Preprocessing:** Encodes categorical variables and scales numerical features dynamically for accurate model inference.

---

## 🛠️ Tech Stack & Tools

* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn (Logistic Regression / Random Forest)
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Web Framework & UI:** Streamlit

---

## ⚙️ How It Works

1. **Data Ingestion & Cleaning:** Raw customer data is processed, handling missing values and encoding categorical attributes (e.g., Contract Type, Internet Service).
2. **Model Training:** Classification algorithms are trained on historical customer churn data to identify predictive patterns.
3. **Inference Pipeline:** User inputs from the Streamlit UI pass through the pre-trained pipeline to output probability scores and risk categorization.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/customer-churn-prediction.git](https://github.com/your-username/customer-churn-prediction.git)
cd customer-churn-prediction
2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
     source venv/bin/activate
3. Install required dependencies
     pip install -r requirements.txt
📌 Project Structure
├── data/                  # Sample datasets / test data
├── models/                # Trained model artifacts (.pkl or .joblib)
├── app.py                 # Main Streamlit web application
├── requirements.txt       # List of Python dependencies
└── README.md              # Project documentation
👤 Author
Portfolio / GitHub: [https://github.com/Sadaf144/telco-customer-churn-ml]

LinkedIn: [(https://www.linkedin.com/in/sadaf-m-dafii/?isSelfProfile=true)]
