import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt


# configure our environment
st.set_page_config(page_title='Customer Intent Purchase Prediction', layout='centered')

with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

def consumer_intent():
    st.title('Consumer Electronics Purchase Intent Prediction')
    st.write('My Model GUI')

    st.markdown("""
    ### **Encoded Values Guide**
    - **Customer Gender**: 0 = Male, 1 = Female  
    - **Product Categories**: 0 = Headphones, 1 = Laptops, 2 = Smart Watches, 3 = Smartphones, 4 = Tablets  
    - **Product Brands**: 0 = Apple, 1 = HP, 2 = Other Brands, 3 = Samsung, 4 = Sony  
    """)
    # input features ProductPrice'), 'CustomerAge'), 'CustomerGender'), 'PurchaseFrequency'),
    # 'CustomerSatisfaction'), 'product_cat_enc'), 'product_brand_enc'), 'cust_age_class'), 'age_gender'), 'age_customerSatisfaction

    # Input fields
    product_price = st.number_input("Product Price", min_value=100.0, max_value=3000.0, step=0.1)
    customer_age = st.slider("Customer Age", 18, 69, 30)
    customer_gender = st.selectbox("Customer Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    purchase_frequency = st.slider("Purchase Frequency", 1, 19, 5)
    customer_satisfaction = st.slider("Customer Satisfaction", 1, 5, 3)
    product_category = st.selectbox("Product Category",
                                [("Headphones (0)", 0), ("Laptops (1)", 1),
                                 ("Smart Watches (2)", 2), ("Smartphones (3)", 3),
                                 ("Tablets (4)", 4)], format_func=lambda x: x[0])[1]
    product_brand = st.selectbox("Product Brand",
                             [("Apple (0)", 0), ("HP (1)", 1), ("Other Brands (2)", 2),
                              ("Samsung (3)", 3), ("Sony (4)", 4)], format_func=lambda x: x[0])[1]

    # Feature engineering (same as in training)
    cust_age_class = 1 if customer_age <= 38 else (2 if customer_age <= 49 else (3 if customer_age <= 59 else 4))
    age_gender = customer_age * customer_gender
    age_customer_satisfaction = customer_age * customer_satisfaction

    # Predict
    features = np.array([[product_price, customer_age, customer_gender, purchase_frequency,
                          customer_satisfaction, product_category, product_brand,
                          cust_age_class, age_gender, age_customer_satisfaction]])

    st.caption("Select appropriate values based on the guide above.")

    # predicted value
    if st.button('Predict'):
        prediction = model.predict(features)

        result = "Likely to Purchase" if prediction[0] == 1 else "Unlikely to Purchase"
        st.write(f"Prediction: **{result}**")

        st.subheader("Feature Contribution (SHAP)")
        explainer = shap.Explainer(model)  # Create SHAP explained
        shap_values = explainer(features)  # Get SHAP values for input

        st.write("SHAP values shape:", shap_values.values.shape)

        # For multi-class models, we need to select one class's explanation.
        # Here, we select class 0 for the first instance.
        # Construct a new Explanation object with the appropriate values.
        shap_value_single = shap.Explanation(
            values=shap_values.values[0, :, 0],  # Shape: (10,)
            base_values=shap_values.base_values[0, 0],  # A scalar base value for class 0
            data=shap_values.data[0],  # The feature values for the instance
            feature_names=shap_values.feature_names if hasattr(shap_values, 'feature_names') else None
        )

        st.write("#### SHAP Waterfall Plot")
        fig, ax = plt.subplots(figsize=(8, 6))  # Create Matplotlib figure
        shap.plots.waterfall(shap_value_single, show=False)  # Generate SHAP plot without auto-display
        st.pyplot(fig)  # Use Streamlit to display the figure

        st.caption("SHAP explains how much each feature influenced the prediction.")

if __name__ == '__main__':
    consumer_intent()