import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image


# here we define some of the front end elements of the web page like the font and background color,
# the padding and the text to be displayed

html_temp = """
	<div style ="background-color:#3d2fd6; padding:13px">
	<h1 style ="color:#f0f0f5; text-align:center; ">Streamlit Iris Flower Classifier </h1>
	</div>
	"""
# this line allows us to display the front end aspects we have defined in the above code
st.markdown(html_temp, unsafe_allow_html = True)

# İmages of Iris Flowers
image = Image.open("iris.jpg")
st.image(image, use_column_width=True)


# Display Iris Dataset
st.header("_Iris Dataset_")
df = pd.read_csv('iris.csv')
st.write(df.head())

# Loading the models to make predictions
log_model = pickle.load(open("final_logistic_pipe_model", "rb"))
rf_model = pickle.load(open("final_RF_model", "rb")) 


# User input variables that will be used on predictions
st.sidebar.title("_Please Enter the Features and Model Name to Predict the Species of Iris Flower_")
model_selected = st.sidebar.selectbox('Select the model', ('Logistic Regression', 'Random Forest'))
sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.8, 0.1)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.05, 0.1)
petal_length = st.sidebar.number_input("Petal Length", 1.0, 6.9, 3.75, 0.1)
petal_width = st.sidebar.number_input("Petal Width", 0.1, 2.5, 1.2, 0.1)


# Converting user inputs to dataframe 
my_dict = {
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    'petal_width':petal_width    
}
df_input = pd.DataFrame.from_dict([my_dict])


# defining the function which will make the prediction using the data
def prediction(model, input_data):
	prediction = model.predict(input_data)
	return prediction
	
	
# Making prediction and displaying results
if st.button("Predict"):
    if model_selected == "Logistic Regression":
        result = prediction(log_model, df_input)[0]
    else :
        result = prediction(rf_model, df_input)[0]

try:
    st.success(f"Iris Flower is **{result}**")
    if result == "versicolor":
    	st.image(Image.open("Iris_versicolor.jpeg"))
    elif result == "setosa":
        st.image(Image.open("Iris_setosa.jpeg"))
    elif result == "virginica":
        st.image(Image.open("Iris_virginica.jpeg"))
except NameError:
    st.write("Please **Predict** button to display the result!")




    


