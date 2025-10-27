import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="AutoML with PyCaret", layout="wide")
st.title("🤖 AutoML Dashboard with PyCaret")

# ======================================================
# 1️⃣ Upload CSV & Select Problem Type
# ======================================================
st.sidebar.header("⚙️ Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    target_col = st.sidebar.selectbox("🎯 Select Target Column", df.columns)
    
    # Auto-detect problem type
    if df[target_col].dtype == 'object' or df[target_col].nunique() < 20:
        default_type = "Classification"
    else:
        default_type = "Regression"
    
    problem_type = st.sidebar.radio(
        "📊 Problem Type",
        ["Classification", "Regression"],
        index=0 if default_type == "Classification" else 1
    )
    
    train_button = st.sidebar.button("🚀 Train Model")

    # ======================================================
    # 2️⃣ Train Model (Dynamic Import)
    # ======================================================
    if train_button:
        try:
            with st.spinner("Training model... This may take a while ⏳"):
                # Dynamic import based on problem type
                if problem_type == "Classification":
                    from pycaret.classification import setup, compare_models, pull, save_model
                    sort_metric = 'Accuracy'
                else:
                    from pycaret.regression import setup, compare_models, pull, save_model
                    sort_metric = 'R2'
                
                # Setup PyCaret
                s = setup(
                    data=df, 
                    target=target_col, 
                    session_id=123, 
                    log_experiment=False, 
                    verbose=False,
                )
                
                # Compare models
                best_model = compare_models(sort=sort_metric, n_select=1)
                
                # Pull comparison results
                results = pull()
                
                st.success(f"✅ {problem_type} model training complete!")
                st.subheader("🏆 Model Comparison Results")
                st.dataframe(results)

                # Save model with problem type metadata
                save_model(best_model, "best_model")
                
                # Save problem type and target column to a file
                with open("model_metadata.txt", "w") as f:
                    f.write(f"{problem_type}|{target_col}")
                
                st.sidebar.success(f"💾 Model saved as 'best_model.pkl' ({problem_type})")
                
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.info("💡 Tips:\n- Check target column data type\n- Ensure no missing values in target\n- Verify dataset has enough samples")

# ======================================================
# 3️⃣ Load Model & Predict
# ======================================================
st.sidebar.header("🔮 Prediction Section")
predict_mode = st.sidebar.checkbox("Use saved model for prediction")

if predict_mode:
    model_path = "best_model.pkl"
    metadata_path = "model_metadata.txt"
    
    if not os.path.exists(model_path):
        st.warning("⚠️ No saved model found. Please train a model first.")
    else:
        try:
            # Load problem type and target column
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = f.read().strip().split("|")
                    saved_problem_type = metadata[0]
                    target_column_name = metadata[1] if len(metadata) > 1 else "Target"
            else:
                saved_problem_type = "Classification"  # fallback
                target_column_name = "Target"
            
            # Dynamic import for prediction
            if saved_problem_type == "Classification":
                from pycaret.classification import load_model, predict_model
            else:
                from pycaret.regression import load_model, predict_model
            
            # Load model
            loaded = load_model("best_model")
            model = loaded[0] if isinstance(loaded, list) else loaded
            
            st.sidebar.info(f"📌 Model: **{saved_problem_type}**\n\n🎯 Target: **{target_column_name}**")
            
            pred_file = st.sidebar.file_uploader("Upload CSV for Prediction", type=["csv"], key="predict")

            if pred_file:
                try:
                    pred_data = pd.read_csv(pred_file)
                except Exception as e:
                    st.error(f"❌ Failed to read CSV: {e}")
                    st.stop()

                st.subheader("📂 Data to Predict")
                st.dataframe(pred_data.head())

                if not isinstance(pred_data, pd.DataFrame) or pred_data.empty:
                    st.error("❌ Uploaded file is not a valid DataFrame or is empty.")
                    st.stop()

                if st.sidebar.button("🔍 Run Prediction"):
                    with st.spinner("Running predictions..."):
                        try:
                            preds = predict_model(model, data=pred_data)
                            
                            # Rename prediction column to target column name
                            if "prediction_label" in preds.columns:
                                preds = preds.rename(columns={"prediction_label": f"predicted_{target_column_name}"})
                            
                            # Also rename prediction_score if exists (for classification)
                            if "prediction_score" in preds.columns:
                                preds = preds.rename(columns={"prediction_score": f"confidence_{target_column_name}"})
                            
                            st.subheader("📊 Prediction Results")
                            st.dataframe(preds)

                            # Download result
                            csv = preds.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="⬇️ Download Predictions as CSV",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                            st.success("✅ Prediction complete!")

                            # Visualization based on problem type
                            pred_col = f"predicted_{target_column_name}"
                            
                            if saved_problem_type == "Classification" and pred_col in preds.columns:
                                st.subheader("📈 Prediction Distribution")
                                st.bar_chart(preds[pred_col].value_counts())
                            elif saved_problem_type == "Regression" and pred_col in preds.columns:
                                st.subheader("📈 Prediction Distribution")
                                st.line_chart(preds[pred_col])

                        except Exception as e:
                            st.error(f"❌ Prediction failed: {e}")
                            st.info("💡 Make sure the uploaded data has the same features as training data")

                        
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")