import streamlit as st
import pickle
import numpy as np

# import the model with error handling
try:
    pipe = pickle.load(open(r"E:\Sujeet\1.DU MSC\Semester 1\Data Mining\Projeact\Final Submission\App\laptop-price-predictor\laptop-price-predictor\pipe.pkl",'rb'))
    df = pickle.load(open(r"E:\Sujeet\1.DU MSC\Semester 1\Data Mining\Projeact\Final Submission\App\laptop-price-predictor\laptop-price-predictor\df.pkl",'rb'))
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("Please upgrade scikit-learn to version 1.6.1 or retrain the model with your current version.")
    model_loaded = False

st.title("Laptop Predictor")

if model_loaded:
    # brand
    company = st.selectbox('Brand',df['Company'].unique())

    # type of laptop
    type = st.selectbox('Type',df['TypeName'].unique())

    # Ram
    ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

    # weight
    weight = st.number_input('Weight of the Laptop')

    # Touchscreen
    touchscreen = st.selectbox('Touchscreen',['No','Yes'])

    # IPS
    ips = st.selectbox('IPS',['No','Yes'])

    # screen size
    screen_size = st.slider('Screensize in inches', 10.0, 18.0, 13.0)

    # resolution
    resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

    #cpu
    cpu = st.selectbox('CPU',df['Cpu brand'].unique())

    hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

    ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

    gpu = st.selectbox('GPU',df['Gpu brand'].unique())

    os = st.selectbox('OS',df['os'].unique())

    if st.button('Predict Price'):
        try:
            # query
            ppi = None
            if touchscreen == 'Yes':
                touchscreen = 1
            else:
                touchscreen = 0

            if ips == 'Yes':
                ips = 1
            else:
                ips = 0

            X_res = int(resolution.split('x')[0])
            Y_res = int(resolution.split('x')[1])
            ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
            query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

            query = query.reshape(1,12)
            predicted_price = int(np.exp(pipe.predict(query)[0]))
            st.title(f"The predicted price of this configuration is ${predicted_price:,}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
else:
    st.warning("Model not loaded. Please fix the version compatibility issue first.")