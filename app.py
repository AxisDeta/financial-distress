import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import joblib
import os
import plotly.express as px
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Financial Risk Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }
    div.css-12w0qpk.e1tzin5v1 {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e3eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #fafafa;
        transition: all 0.3s ease;
    }
    .upload-box:hover {
        border-color: #4CAF50;
        background: #f8f9fa;
    }
    footer {
        margin-top: 3rem;
        padding: 0.5rem;
        text-align: center;
        font-size: 0.8rem;
        color: #666;
        border-top: 1px solid #eee;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .prediction-card h3 {
        color: #1a237e;
        margin-bottom: 1rem;
    }
    .prediction-card p {
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    .prediction-card strong {
        color: #1a237e;
    }
    h1 {
        color: #1f1f1f;
        font-weight: bold;
    }
    h2, h3 {
        color: #2c3e50;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Paths for data and models
DATA_PATH = 'financial_disaster_dataset.csv'
BINARY_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_binary_model.pkl')
MULTICLASS_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_multiclass_model.pkl')
CALIBRATED_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'calibrated_binary_model.pkl')

# Load models
try:
    best_binary_model = joblib.load(BINARY_MODEL_PATH)
    best_multiclass_model = joblib.load(MULTICLASS_MODEL_PATH)
    if os.path.exists(CALIBRATED_MODEL_PATH):
        calibrated_binary_model = joblib.load(CALIBRATED_MODEL_PATH)
    else:
        calibrated_binary_model = None
    st.sidebar.success("✅ Models loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading models: {str(e)}")
    best_binary_model = None
    best_multiclass_model = None
    calibrated_binary_model = None

# Load the dataset and perform preprocessing

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data.sort_values(by=['Family_ID', 'Round'], inplace=True)

    disaster_level_order = [['Low', 'Medium', 'High']]
    encoder = OrdinalEncoder(categories=disaster_level_order)
    data['Disaster_Level_Encoded'] = encoder.fit_transform(data[['Disaster_Level']])

    numerical_features_for_ts = data.select_dtypes(include=np.number).columns.tolist()
    numerical_features_for_ts.remove('Family_ID')
    numerical_features_for_ts.remove('Round')
    numerical_features_for_ts.remove('Label')
    numerical_features_for_ts.remove('Disaster_Level_Encoded')

    for feature in numerical_features_for_ts:
        data[f'{feature}_lag1'] = data.groupby('Family_ID')[feature].shift(1).fillna(0)
        data[f'{feature}_rolling_mean'] = data.groupby('Family_ID')[feature].expanding().mean().reset_index(level=0, drop=True)
        data[f'{feature}_rolling_std'] = data.groupby('Family_ID')[feature].expanding().std().reset_index(level=0, drop=True).fillna(0)
        data[f'{feature}_velocity'] = data.groupby('Family_ID')[feature].diff(1).fillna(0)

    data['Disaster_Borrowing_Interaction'] = data['Natural_Disaster_Impact'] * data['Household_Borrowing_Rate']
    data['Household_Borrowing_Rate_Sq'] = data['Household_Borrowing_Rate']**2

    leakage_features = [col for col in data.columns if 'Disaster_Level' in col and col != 'Disaster_Level_Encoded']
    corrected_feature_columns = [col for col in data.columns if col not in ['Label', 'Disaster_Level_Encoded', 'Family_ID', 'Round'] + leakage_features]

    modeling_data = data[corrected_feature_columns + ['Label', 'Disaster_Level_Encoded', 'Family_ID', 'Round']].copy()


    return modeling_data, corrected_feature_columns

data, feature_columns = load_data(DATA_PATH)

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/fluency/96/analytics.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Make Prediction", "Data Explorer"])

# Add research link in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 Research Details
[View the feature engineering methodology and research details](https://colab.research.google.com/drive/1x50MyE72lglYd-w2iCbJQqJvtM4N_Yqs?usp=sharing)
""")

if page == "Home":
    st.title("🏦 Financial Distress Prediction System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to the Financial Risk Assessment Platform
        
        This intelligent system helps predict:
        - 📊 Likelihood of financial distress
        - 🌪️ Potential disaster impact levels
        - 💡 Risk mitigation suggestions
        
        **How it works:**
        1. Input your financial indicators
        2. Get instant AI-powered predictions
        3. Receive detailed risk analysis
        
        **Want to learn more?**  
        📚 [View the research and feature engineering details](https://colab.research.google.com/drive/1x50MyE72lglYd-w2iCbJQqJvtM4N_Yqs?usp=sharing)
        """)
    
    with col2:
        # Using a base64 encoded SVG for reliability
        st.markdown("""
            <svg width="300" height="300" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 44H44" stroke="#4CAF50" stroke-width="2" stroke-linecap="round"/>
                <path d="M4 4V44" stroke="#4CAF50" stroke-width="2" stroke-linecap="round"/>
                <path d="M44 44V26" stroke="#4CAF50" stroke-width="2" stroke-linecap="round"/>
                <path d="M4 35L14 25L24 35L44 15" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M34 15H44V25" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            """, unsafe_allow_html=True)

elif page == "Make Prediction":
    st.title("📈 Risk Assessment Calculator")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["✍️ Manual Input", "📤 Batch Upload"])
    
    with tab1:
        with st.expander("📝 Input Guidelines", expanded=True):
            st.info("Fill in the financial indicators below. Hover over each field for more information.")
        
        input_data = {}
        with st.form("prediction_form", clear_on_submit=False):
            # Organize features into logical groups
            economic_features = [f for f in feature_columns if 'Economic' in f or 'Financial' in f or 'Income' in f]
            disaster_features = [f for f in feature_columns if 'Disaster' in f or 'Risk' in f]
            household_features = [f for f in feature_columns if 'Household' in f or 'Social' in f]
            
            # Economic Indicators Section
            st.markdown("### 💰 Economic Indicators")
            col1, col2, col3 = st.columns(3)
            for i, feat in enumerate(economic_features):
                with eval(f"col{(i % 3) + 1}"):
                    default_value = float(data[feat].mean()) if feat in data.columns else 0.0
                    input_data[feat] = st.number_input(
                        feat.replace('_', ' ').title(),
                        value=default_value,
                        format="%.2f",
                        help=f"Enter the value for {feat.replace('_', ' ').lower()}"
                    )
            
            # Disaster Risk Section
            st.markdown("### 🌪️ Disaster Risk Factors")
            col1, col2, col3 = st.columns(3)
            for i, feat in enumerate(disaster_features):
                with eval(f"col{(i % 3) + 1}"):
                    default_value = float(data[feat].mean()) if feat in data.columns else 0.0
                    input_data[feat] = st.number_input(
                        feat.replace('_', ' ').title(),
                        value=default_value,
                        format="%.2f",
                        help=f"Enter the value for {feat.replace('_', ' ').lower()}"
                    )
            
            # Other remaining features
            remaining_features = [f for f in feature_columns if f not in economic_features + disaster_features]
            if remaining_features:
                st.markdown("### 📊 Additional Indicators")
                col1, col2, col3 = st.columns(3)
                for i, feat in enumerate(remaining_features):
                    with eval(f"col{(i % 3) + 1}"):
                        default_value = float(data[feat].mean()) if feat in data.columns else 0.0
                        input_data[feat] = st.number_input(
                            feat.replace('_', ' ').title(),
                            value=default_value,
                            format="%.2f",
                            help=f"Enter the value for {feat.replace('_', ' ').lower()}"
                        )
            
            submitted = st.form_submit_button("🔍 Analyze Risk Profile")
    
    with tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 📤 Batch Processing")
            st.write("""
            1. Download the template CSV file
            2. Fill it with your data (one row per prediction)
            3. Upload the completed file
            """)
        
        st.markdown("---")
        
        # Create template DataFrame with feature columns
        template_df = pd.DataFrame(columns=feature_columns)
        template_df.loc[0] = [0.0] * len(feature_columns)  # Add one row with zeros as example
        
        # Add download button for template
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_template,
            file_name="prediction_template.csv",
            mime="text/csv",
            help="Download a CSV template with all required columns"
        )
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                with st.spinner('Processing batch data...'):
                    # Read and validate the uploaded file
                    batch_data = pd.read_csv(uploaded_file)
                    
                    # Check if all required columns are present
                    missing_cols = set(feature_columns) - set(batch_data.columns)
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        # Make batch predictions
                        if best_binary_model is not None and best_multiclass_model is not None:
                            batch_binary_proba = best_binary_model.predict_proba(batch_data[feature_columns])[:, 1]
                            batch_multiclass_pred = best_multiclass_model.predict(batch_data[feature_columns])
                            
                            # Add predictions to the dataframe
                            results_df = batch_data.copy()
                            results_df['Distress_Probability'] = batch_binary_proba
                            results_df['Predicted_Risk_Level'] = [disaster_level_mapping.get(x, 'Unknown') for x in batch_multiclass_pred]
                            
                            # Show preview of results
                            st.success('✅ Batch processing complete!')
                            st.write("Preview of results:")
                            st.dataframe(results_df.head())
                            
                            # Download button for results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )
                with st.spinner('Processing batch data...'):
                    # Read and validate the uploaded file
                    batch_data = pd.read_csv(uploaded_file)
                    
                    # Check if all required columns are present
                    missing_cols = set(feature_columns) - set(batch_data.columns)
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        # Make batch predictions
                        if best_binary_model is not None and best_multiclass_model is not None:
                            batch_binary_proba = best_binary_model.predict_proba(batch_data[feature_columns])[:, 1]
                            batch_multiclass_pred = best_multiclass_model.predict(batch_data[feature_columns])
                            
                            # Add predictions to the dataframe
                            results_df = batch_data.copy()
                            results_df['Distress_Probability'] = batch_binary_proba
                            results_df['Predicted_Risk_Level'] = [disaster_level_mapping.get(x, 'Unknown') for x in batch_multiclass_pred]
                            
                            # Show preview of results
                            st.success('✅ Batch processing complete!')
                            st.write("Preview of results:")
                            st.dataframe(results_df.head())
                            
                            # Download button for results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV file is properly formatted and contains all required columns.")

    # --- Prediction Results ---
    if submitted:
        if best_binary_model is not None and best_multiclass_model is not None:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data], columns=feature_columns)

            if list(input_df.columns) != feature_columns:
                st.error("Input data columns do not match the expected feature columns.")
            else:
                try:
                    # Make predictions
                    binary_proba = calibrated_binary_model.predict_proba(input_df)[:, 1] if calibrated_binary_model else best_binary_model.predict_proba(input_df)[:, 1]
                    binary_prediction = (binary_proba > 0.5).astype(int)[0]
                    
                    multiclass_proba = best_multiclass_model.predict_proba(input_df)[0]
                    multiclass_prediction_encoded = best_multiclass_model.predict(input_df)[0]
                    
                    disaster_level_mapping = {0.0: 'Low', 1.0: 'Medium', 2.0: 'High'}
                    multiclass_prediction_label = disaster_level_mapping.get(multiclass_prediction_encoded, 'Unknown')

                    # Display results in a modern card layout
                    st.markdown("### 📊 Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        <div class='prediction-card'>
                            <h3>Financial Distress Analysis</h3>
                            <p><strong>Status:</strong> %s</p>
                            <p><strong>Confidence:</strong> %.1f%%</p>
                        </div>
                        """ % ('⚠️ High Risk' if binary_prediction == 1 else '✅ Low Risk', 
                               binary_proba[0] * 100), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class='prediction-card'>
                            <h3>Disaster Risk Level</h3>
                            <p><strong>Predicted Level:</strong> %s</p>
                            <p><strong>Confidence:</strong> %.1f%%</p>
                        </div>
                        """ % (multiclass_prediction_label, 
                               max(multiclass_proba) * 100), unsafe_allow_html=True)
                    
                    # Add risk mitigation suggestions
                    st.markdown("### 💡 Risk Mitigation Suggestions")
                    
                    if binary_prediction == 1:  # High Risk
                        st.warning("""
                        Based on our analysis, here are the recommended actions:
                        - Review and potentially restructure current borrowing arrangements
                        - Implement stricter financial monitoring and controls
                        - Consider building additional emergency reserves
                        - Evaluate and adjust disaster preparedness plans
                        """)
                    else:  # Low Risk
                        st.success("""
                        While the current risk is low, consider these preventive measures:
                        - Maintain current financial management practices
                        - Continue regular risk assessments
                        - Build upon existing emergency funds
                        - Keep disaster response plans updated
                        """)

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.error("Please ensure the input values are valid.")
        else:
            st.warning("Models not loaded. Please ensure model files exist at the specified paths.")

elif page == "Data Explorer":
    st.title("🔍 Data Explorer")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution Analysis", "🔄 Time Series Patterns", "🔗 Correlation Analysis", "📈 Feature Importance"])
    
    with tab1:
        if st.checkbox("Show Dataset Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Numerical Features Distribution")
                feature_to_plot = st.selectbox("Select Feature", feature_columns)
                fig = px.histogram(data, x=feature_to_plot, nbins=30)
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 Feature Statistics")
                st.dataframe(data[feature_columns].describe())

        st.subheader("Disaster Level Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(data, x='Disaster_Level_Encoded', title='Distribution of Disaster Level')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(data, x='Label', title='Distribution of Binary Label')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Time Series Patterns")
        # Sample a few families for visualization
        sample_size = st.slider("Number of families to display", 1, 10, 3)
        sample_family_ids = data['Family_ID'].unique()[:sample_size]
        
        time_series_features = ['GDP_Growth', 'Inflation', 'Market_Liquidity', 'ICT_Demand', 
                              'CyberIncident_Count', 'Household_Borrowing_Rate']
        feature_to_plot = st.selectbox("Select Feature for Time Series", time_series_features)
        
        fig = px.line(data[data['Family_ID'].isin(sample_family_ids)], 
                     x='Round', y=feature_to_plot, color='Family_ID',
                     title=f'Time Series of {feature_to_plot} for Sample Families')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Correlation Analysis")
        # Compute correlation matrix for selected features
        correlation_features = st.multiselect(
            "Select features for correlation analysis",
            feature_columns,
            default=feature_columns[:10]
        )
        
        if correlation_features:
            correlation_matrix = data[correlation_features].corr()
            fig = px.imshow(correlation_matrix, 
                          title="Feature Correlation Matrix",
                          aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Feature Importance")
        if best_binary_model is not None and hasattr(best_binary_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': best_binary_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        title='Feature Importance from Binary Model',
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance visualization is only available for tree-based models after they are loaded.")

# Add footer
st.markdown("""
<footer>
    <p>Financial Risk Analyzer v1.0 | Created with Streamlit | Last Updated: {}
    <br>
    <small>© 2024 Financial Risk Analyzer. For questions or support, please contact <a href="mailto:axisdeta@gmail.com" style="color: #4CAF50; text-decoration: none;">axisdeta@gmail.com</a></small>
    </p>
</footer>
""".format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
