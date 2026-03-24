# 🏠 House Price Prediction

## 📌 Overview

This project builds a machine learning model to predict house prices based on various socio-economic and housing features. The goal is to understand key factors influencing real estate pricing and develop an end-to-end regression pipeline.

---

## ⚙️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Model:** Random Forest Regressor

---

## 📂 Dataset

* Dataset contains features such as:

  * Avg. Area Income
  * Avg. Area House Age
  * Avg. Area Number of Rooms
  * Avg. Area Number of Bedrooms
  * Area Population
  * Price (Target Variable)
* Non-numeric column **Address** was removed during preprocessing

---

## 🔍 Approach

### 1. Data Preprocessing

* Removed non-numeric column (`Address`)
* Checked for missing values and handled appropriately

### 2. Exploratory Data Analysis (EDA)

* Analyzed distributions of features
* Generated correlation heatmap
* Visualized relationships between features and target variable

### 3. Feature Engineering

* Selected relevant numerical features
* Applied feature scaling using StandardScaler

### 4. Model Training

* Split dataset into train/test sets
* Trained a **Random Forest Regressor**

### 5. Evaluation

* Evaluated model using:

  * RMSE (Root Mean Squared Error)
  * R² Score

---

## 📊 Key Insights

* **Avg. Area Income** is the strongest predictor of house prices
* Number of rooms significantly influences property value
* Population has moderate impact on pricing trends
* Bedrooms alone are less influential than total rooms
* Address feature was removed as it is non-numeric

---

## 🚀 Results

* Achieved strong predictive performance using Random Forest
* RMSE and R² indicate good model fit (values may vary based on dataset)

  * Performance metrics:
  * RMSE: 119,970
  * R2 Score: 0.883 (the model explains ~88% of the variance in house prices)

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py
```

---

## 📁 Project Structure

```
house-price-prediction/
│
├── data/                # Dataset
├── notebooks/           # EDA notebook
├── src/                 # Training & inference scripts
├── requirements.txt
└── README.md
|__ model.pkl            # Both the trained model and the StandardScaler are pickled together
```

---

## 🔮 Future Improvements

* Add feature importance visualization
* Try advanced models (XGBoost, Gradient Boosting)
* Deploy model using Streamlit or Flask
* Perform hyperparameter tuning

---

## 👤 Author

Naveen Reddy CH
📍 Hyderabad, India
📧 [naveenreddych123@gmail.com](mailto:naveenreddych123@gmail.com)
