# **Consumer Electronics Sales - Customer Intent Prediction**

## 📌 **Project Overview**
This project predicts whether a customer is likely to purchase a consumer electronics product based on various factors such as **product price, customer age, gender, purchase frequency, satisfaction level, and product category/brand**.

## Live Link


A **Random Forest model** was trained to make predictions, and the model was deployed using **Streamlit** for easy interaction. Additionally, **SHAP (SHapley Additive Explanations)** was integrated to interpret model predictions.

---

## 🚀 **Features**
✔ **Data Preprocessing**: Encoding categorical variables, scaling numerical features, handling missing values, and removing outliers.  
✔ **Feature Engineering**: Creating new features such as customer age class and interactions between age, gender, and satisfaction.  
✔ **Model Training & Evaluation**: Comparing multiple machine learning models (Logistic Regression, Decision Trees, Random Forest) and handling class imbalance with **SMOTE**.  
✔ **Model Deployment with Streamlit**: A user-friendly web app that allows users to input customer details and get real-time predictions.  
✔ **Explainability with SHAP**: Visualizing how each feature influences predictions using **waterfall plots**.  

---

## 📂 **Project Structure**
```
📦 Consumer-Intent-Prediction
│-- 📂 data/                      # Dataset (CSV files)
│-- 📂 models/                    # Saved ML model (rf_model.pkl)
│-- 📂 notebooks/                 # Jupyter notebooks for EDA & model training
│-- 📜 app.py                      # Streamlit app for deployment
│-- 📜 requirements.txt            # Dependencies
│-- 📜 README.md                   # Project documentation (this file)
```

---

## ⚙ **Installation & Setup**

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-repo/Consumer-Intent-Prediction.git
cd Consumer-Intent-Prediction
```

### 2️⃣ **Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv env
source env/bin/activate  # On Mac/Linux
env\Scripts\activate  # On Windows
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Run the Streamlit App**
```bash
streamlit run app.py
```

---

## 🔍 **How to Use the App**
1️⃣ Enter customer details such as **product price, age, gender, purchase frequency, and satisfaction level**.  
2️⃣ Select the **product category** and **brand**.  
3️⃣ Click on **"Predict"** to check if the customer is likely to purchase.  
4️⃣ View the **SHAP Waterfall Plot** to understand **why the model made that prediction**.  

---

## 🛠 **Technologies Used**
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)  
- **Machine Learning** (Scikit-learn, SMOTE for imbalance handling)  
- **SHAP** (Model interpretability)  
- **Streamlit** (Model deployment)  

---

## 🎯 **Next Steps**
📌 Improve model performance using **hyperparameter tuning**.  
📌 Add **more visualizations** to the Streamlit app.  


## 🤝 **Contributing**
Contributions are welcome! Feel free to **fork the repo**, create a **pull request**, or open an **issue**.  

---

## 📞 **Contact**
👤 **Favour Ezekwu**  
📧 Email: fayvehz@gmail.com  
🔗 LinkedIn: linkedin.com/in/favourezekwu

