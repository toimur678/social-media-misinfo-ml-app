# Social Media Misinformation Detection Web App

## Overview
This project is a **Machine Learning-powered web application** built with **Streamlit** that predicts whether a user is likely to spread misinformation on social media based on their demographic and platform usage. The app uses a trained machine learning model to make predictions and provides an interactive interface for users to input data and view results.

---

## Features
- **User Input**: Users can input details such as:
  - Social media platform (e.g., Facebook, Twitter)
  - Country
  - Age
  - Gender
- **Prediction**: The app predicts whether the user is likely to spread misinformation or not.
- **Dynamic Preprocessing**: The app dynamically preprocesses user input to match the model's requirements.
- **Visual Feedback**: Results are displayed with color-coded feedback (red for "Spreads misinformation" and green for "Does not spread misinformation").

---

## Technologies Used
- **Python**: Core programming language.
- **Streamlit**: For building the web app interface.
- **Scikit-learn**: For machine learning model training and evaluation.
- **Pandas**: For data manipulation and preprocessing.
- **Joblib**: For saving and loading the trained model.
- **NumPy**: For data generation and numerical operations.

---

## Dataset
The dataset used for training the model is generated using a Python script (`data.ipynb`). It includes the following features:
- **User_ID**: Unique identifier for each user.
- **Platform**: Social media platform (e.g., Facebook, Twitter).
- **Country**: User's country of residence.
- **Age**: User's age.
- **Gender**: User's gender (Male, Female, Other).
- **Post_Type**: Type of post (Text, Image, Video, Mixed).
- **Engagements**: Number of engagements (likes, shares, etc.).
- **Misinformation_Spread**: Binary target variable (0 = No, 1 = Yes).

---

## Machine Learning Models
The app uses a **Random Forest Classifier** (or the best-performing model) trained on the dataset. Other models evaluated during development include:
- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Naive Bayes
- K-Nearest Neighbors (KNN)
- AdaBoost
- Gradient Boosting

---

## How to Run the App

### Prerequisites
- Python 3.8 or higher
- Required Python libraries: `streamlit`, `pandas`, `scikit-learn`, `joblib`, `numpy`

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/toimur678-social-media-misinfo-ml-app.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd toimur678-social-media-misinfo-ml-app
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(If you don't have a `requirements.txt`, install the libraries manually: `pip install streamlit pandas scikit-learn joblib numpy`)*

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
5. **Access the app**:
   Open your browser and go to `http://localhost:8501` to interact with the app.

---

## File Structure
```
toimur678-social-media-misinfo-ml-app/
├── README.md               # Project documentation
├── algorithm.ipynb         # Jupyter notebook for model training
├── app.py                  # Streamlit web app
├── best_model.joblib       # Trained machine learning model
├── data.csv                # Generated dataset
└── data.ipynb              # Script to generate the dataset
```

---

## Customization
- **Dataset Generation**: Modify `data.ipynb` to generate a custom dataset with different platforms, countries, or age ranges.
- **Model Training**: Use `algorithm.ipynb` to experiment with different machine learning models or hyperparameters.
- **App Interface**: Customize the app's interface by editing `app.py`.

---

## Acknowledgments
- **Streamlit** for providing an easy-to-use framework for building web apps.
- **Scikit-learn** for its comprehensive machine learning tools.
- **Pandas** and **NumPy** for data manipulation and numerical operations.

---

**Note**: This app is for educational and demonstration purposes only. The predictions are based on synthetic data and may not reflect real-world scenarios.
