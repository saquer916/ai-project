"""
Streamlit app:  train model, evaluate, backtest, deploy.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_generator import generate_training_dataset
from src.features import feature_engineering
from src.model import XGBoostModel
from src.backtest import compare_strategies
from src.inference import PricingAgent
import os

st.set_page_config(page_title="ML Stock Pricing Signals", layout="wide")
st.title("ML-Powered Stock Pricing Signals for SMEs")

# Sidebar for mode selection
mode = st.sidebar. radio("Select Mode", ["Train Model", "Evaluate & Backtest", "Real-time Inference"])

# Initialize session state
if 'model' not in st.session_state:
    st.session_state. model = None
if 'df_train' not in st.session_state:
    st.session_state.df_train = None
if 'features_to_use' not in st.session_state:
    st.session_state.features_to_use = None

# ==================== MODE 1: TRAIN ====================
if mode == "Train Model":
    st.header("Train Pricing Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("Training samples", 100, 2000, 500, 100)
    with col2:
        random_seed = st.slider("Random seed", 0, 100, 42)
    
    if st.button("Generate Training Data & Train Model", key="train_btn"):
        with st.spinner("Generating training data..."):
            df = generate_training_dataset(n_samples=n_samples, seed=random_seed)
            st.session_state.df_train = df
            st.write(f"Generated {len(df)} samples")
            
            # Show sample
            st.subheader("Sample Data")
            st.dataframe(df.head(10))
            
            st.subheader("Data Statistics")
            st.write(df.describe())
        
        with st.spinner("Engineering features..."):
            df_feat, feat_names = feature_engineering(df)
            st.session_state.features_to_use = feat_names
            st.write(f"Created {len(feat_names)} features")
            st.write(feat_names)
        
        with st.spinner("Training XGBoost model..."):
            model = XGBoostModel()
            # Correct usage
            X = df_feat[feat_names]
            y = df_feat['profit_impact']   # target column
            metrics = model.train(X, y)
            st.session_state. model = model
            
            st.subheader("Training Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Test RMSE", f"{metrics['test_rmse']:.3f}")
            col2.metric("Test MAE", f"{metrics['test_mae']:.3f}")
            col3.metric("Test R²", f"{metrics['test_r2']:.3f}")
            col4.metric("Train R²", f"{metrics['train_r2']:.3f}")
            
            # Feature importance
            st.subheader("Feature Importance")
            df_imp = model.feature_importance()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df_imp. head(15), x='importance', y='feature', ax=ax)
            ax.set_title("Top 15 Most Important Features")
            st.pyplot(fig)
            
            # Save model
            if not os.path.exists("models"):
                os.makedirs("models")

            # Save with a proper filename inside the folder
            model.save("models/xgb_model.pkl")
            st.success("Model trained and saved!")


# ==================== MODE 2: EVALUATE & BACKTEST ====================
elif mode == "Evaluate & Backtest": 
    st.header("Backtest & Evaluate Model")
    
    if st.session_state.model is None or st.session_state.df_train is None:
        st. warning("Train a model first!")
    else:
        df = st.session_state.df_train
        model = st.session_state.model
        features = st.session_state.features_to_use
        
        st.subheader("Model Performance Summary")
        metrics = model.metadata['metrics']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test RMSE", f"{metrics['test_rmse']:.3f}")
        col2.metric("Test MAE", f"{metrics['test_mae']:.3f}")
        col3.metric("Test R²", f"{metrics['test_r2']:.3f}")
        col4.metric("Samples", f"{metrics['n_samples_test']}")
        
        # Backtest
        st.subheader("Backtest Comparison:  ML Model vs Rule Baseline")
        comparison = compare_strategies(model, features)
        
        col1, col2 = st. columns(2)
        
        with col1:
            st.write("**ML Model Strategy**")
            st.json(comparison['model'])
        
        with col2:
            st.write("**Rule Baseline Strategy**")
            st.json(comparison['rule_baseline'])
        
        improvement = comparison['improvement_pct']
        st. metric("Improvement over Baseline", f"{improvement:.1f}%", 
                  delta=improvement, delta_color="normal" if improvement > 0 else "inverse")
        
        # Prediction vs Actual
        st.subheader("Prediction Distribution")
        df_feat, _ = feature_engineering(df)
        y_pred = model.predict(df_feat[features])
        y_actual = df['profit_impact']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        
        # Histogram
        ax1.hist(y_actual, bins=30, alpha=0.5, label='Actual', color='blue')
        ax1.hist(y_pred, bins=30, alpha=0.5, label='Predicted', color='orange')
        ax1.set_xlabel('Profit Impact (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.set_title('Actual vs Predicted Profit Impact Distribution')
        
        # Scatter
        ax2.scatter(y_actual, y_pred, alpha=0.5)
        ax2.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
        ax2.set_xlabel('Actual Profit Impact (%)')
        ax2.set_ylabel('Predicted Profit Impact (%)')
        ax2.set_title('Prediction Accuracy')
        
        st.pyplot(fig)

# ==================== MODE 3: INFERENCE ====================
elif mode == "Real-time Inference": 
    st.header("Get Pricing Recommendations")
    
    if st.session_state. model is None:
        st. warning("Train a model first!")
    else:
        agent = PricingAgent(st. session_state.model)
        
        product = st.selectbox(
            "Select product",
            ["steel components", "car parts", "electronics", "chemicals", "textiles"]
        )
        
        if st.button("Analyze & Get Recommendation", key="infer_btn"):
            with st.spinner("Fetching live stock data..."):
                result = agent.analyze_product(product)
            
            if 'error' in result:
                st.error(f"{result['error']}")
            else:
                st.subheader(f"Product: {product. title()}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Price Recommendation",
                    f"{result['price_recommendation_pct']: +.1f}%",
                    delta=result['price_recommendation_pct']
                )
                col2.metric("Confidence", f"{result['confidence']:. 1%}")
                col3.metric("Expected Profit Impact", f"{result['predicted_profit_impact']: +.2f}%")
                
                # Supplier insights
                st.subheader("Supplier Signals")
                for sup in result['suppliers']:
                    col1, col2, col3 = st.columns(3)
                    col1.write(f"**{sup['name']}** ({sup['ticker']})")
                    col2.metric("30d Move", f"{sup['pct_30d']: +.1f}%")
                
                # Customer insights
                st. subheader("Customer Signals")
                for cust in result['customers']:
                    col1, col2, col3 = st.columns(3)
                    col1.write(f"**{cust['name']}** ({cust['ticker']})")
                    col2.metric("30d Move", f"{cust['pct_30d']:+.1f}%")
                
                # Reasoning
                st.subheader("Recommendation Logic")
                st.info(
                    f"Model predicts ${result['predicted_profit_impact']: +.2f}% profit impact.  "
                    f"With {result['confidence']:.0%} confidence, recommend "
                    f"{result['price_recommendation_pct']:+.1f}% price adjustment."
                )