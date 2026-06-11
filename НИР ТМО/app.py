import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Калькулятор страховых выплат")

st.title("Модель МО")

# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv('/Users/vael/Desktop/insurance.csv')

df = load_data()

st.sidebar.header("Параметры обучения")
n_estimators = st.sidebar.slider("Количество деревьев (n_estimators)", 10, 500, 100)
learning_rate = st.sidebar.slider("Скорость обучения (learning_rate)", 0.01, 0.5, 0.1)
max_depth = st.sidebar.slider("Максимальная глубина (max_depth)", 1, 10, 3)

# Подготовка данных
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2026669)

# Построение и обучение модели
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

model = GradientBoostingRegressor(
    n_estimators=n_estimators, 
    learning_rate=learning_rate, 
    max_depth=max_depth,
    random_state=2026669
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

with st.spinner('Обучение модели...'):
    pipeline.fit(X_train, y_train)

# Оценка
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.header("Результаты модели")
col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("R2 Score", f"{r2:.4f}")

# Предсказание для новых данных
st.header("Предсказание страховых выплат")
age = st.number_input("Возраст", 18, 100, 30)
sex = st.selectbox("Пол", ["male", "female"])
bmi = st.number_input("ИМТ (BMI)", 10.0, 60.0, 25.0)
children = st.number_input("Количество детей", 0, 10, 0)
smoker = st.selectbox("Курильщик", ["yes", "no"])
region = st.selectbox("Регион", df['region'].unique())

input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                         columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

prediction = pipeline.predict(input_data)[0]
st.success(f"Прогноз страховых выплат: ${prediction:,.2f}")
