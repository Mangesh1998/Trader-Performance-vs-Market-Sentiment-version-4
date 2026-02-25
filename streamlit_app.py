"""Streamlit dashboard for exploring Trader Performance vs Market Sentiment results."""

from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Trader vs Sentiment Analysis",
    page_icon="📈",
    layout="wide",
)

# Paths
BASE = Path(__file__).resolve().parent
OUTPUTS = BASE / "outputs"
FIGURES = OUTPUTS / "figures"

st.title("📈 Trader Performance vs Market Sentiment")
st.markdown("**Primetrade.ai Data Science Intern Assignment**")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Data Summary", "Analysis Charts", "Segments", "Strategies", "Prediction", "Model Performance"],
)

if page == "Overview":
    st.header("Overview")
    st.markdown("""
    This dashboard explores how **Bitcoin Fear & Greed sentiment** relates to trader behavior and performance on Hyperliquid.
    
    - **Part A**: Data preparation and daily metrics
    - **Part B**: Performance, behavior, and segments by sentiment
    - **Part C**: Actionable strategy ideas
    - **Bonus**: Predictive model for next-day profitability bucket
    """)
    if FIGURES.exists():
        st.info("Run `python run_pipeline.py` to generate all outputs, then refresh this dashboard.")

elif page == "Data Summary":
    st.header("Data Summary")
    daily_path = OUTPUTS / "daily_metrics.csv"
    segments_path = OUTPUTS / "trader_segments.csv"
    
    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        st.subheader("Daily Metrics")
        st.dataframe(daily.head(100), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(daily))
        with col2:
            st.metric("Unique Accounts", daily["Account"].nunique())
        with col3:
            st.metric("Date Range", f"{daily['date'].min()} to {daily['date'].max()}")
    else:
        st.warning("Run the pipeline first to generate daily_metrics.csv")
    
    if segments_path.exists():
        segments = pd.read_csv(segments_path)
        st.subheader("Trader Segments")
        st.dataframe(segments, use_container_width=True)

elif page == "Analysis Charts":
    st.header("Analysis Charts")
    
    charts = [
        ("q1_performance_by_sentiment.png", "Q1: Performance by Sentiment (PnL, Win Rate, Drawdown)"),
        ("q2_behavior_by_sentiment.png", "Q2: Behavior by Sentiment (Frequency, Leverage, Long/Short)"),
        ("q3_segments.png", "Q3: Segment Comparison (Leverage, Frequency, Consistency)"),
        ("insight1_sentiment_over_time.png", "Insight 1: Sentiment Over Time"),
        ("insight2_pnl_distribution.png", "Insight 2: PnL Distribution"),
        ("insight3_correlation.png", "Insight 3: Feature Correlations"),
    ]
    
    for fname, caption in charts:
        path = FIGURES / fname
        if path.exists():
            st.subheader(caption)
            st.image(str(path), use_container_width=True)
        else:
            st.caption(f"{fname} not found — run pipeline to generate.")

elif page == "Segments":
    st.header("Trader Segments")
    segments_path = OUTPUTS / "trader_segments.csv"
    
    if segments_path.exists():
        segments = pd.read_csv(segments_path)
        for col in ["leverage_segment", "frequency_segment", "consistency_segment"]:
            if col in segments.columns:
                st.subheader(col.replace("_", " ").title())
                st.bar_chart(segments.groupby(col)["avg_pnl"].mean())
    else:
        st.warning("Run the pipeline first.")

elif page == "Strategies":
    st.header("Actionable Strategies (Part C)")
    strat_path = OUTPUTS / "actionable_strategies.csv"
    
    if strat_path.exists():
        strategies = pd.read_csv(strat_path)
        for _, row in strategies.iterrows():
            st.markdown(f"### {row.get('name', 'Strategy')}")
            st.info(row.get("rule", ""))
            st.caption(f"Evidence: {row.get('evidence', '')}")
    else:
        st.warning("Run the pipeline to generate strategies.")

    # Model-driven strategy suggestions (if model available)
    models_dir = BASE / "models"
    model_path = models_dir / "model.pkl"
    fi_path = OUTPUTS / "feature_importances.csv"
    if model_path.exists():
        st.subheader("Model-driven Rules of Thumb")
        try:
            import pickle
            saved = pickle.load(open(model_path, "rb"))
            feimp = saved.get("feature_importances") or {}
            # show top features
            if feimp:
                fi_df = pd.DataFrame.from_dict(feimp, orient="index", columns=["importance"]).sort_values("importance", ascending=False)
                top = fi_df.head(5)
                st.markdown("**Top model features**")
                st.table(top)

            st.markdown("**Model-driven strategy ideas**")
            st.markdown("- If the model predicts a high probability of a loss day: reduce leverage by 50%, tighten stops, avoid opening new positions.")
            st.markdown("- For accounts with recent high exposure on top negative features (e.g., high avg_leverage or negative recent pnl), recommend temporary pause or hedging.")
            st.markdown("- If model signals low risk (profit day): consider scaling in conservatively and keep risk per trade limited.")
        except Exception:
            st.caption("Model not loadable for strategy augmentation.")
    else:
        st.info("Model not trained — run the pipeline to enable model-driven strategies.")

elif page == "Prediction":
    st.header("Prediction — Next-Day Profitability")
    daily_path = OUTPUTS / "daily_metrics.csv"
    model_path = BASE / "models" / "model.pkl"

    if not daily_path.exists():
        st.warning("Run the pipeline to generate daily metrics first.")
    elif not model_path.exists():
        st.warning("Train the model by running the pipeline to enable predictions.")
    else:
        try:
            daily = pd.read_csv(daily_path)
            import pickle
            saved = pickle.load(open(model_path, "rb"))
            model = saved.get("model")
            scaler = saved.get("scaler")
            le = saved.get("encoder")
            feature_names = saved.get("feature_names") or []

            # Selection UI: pick a row from daily metrics
            labels = [f"{row['Account']} | {row['date']} | idx:{int(row['index'])}" for _, row in daily.reset_index().iterrows()]
            # fallback simpler index list if above fails
            if not labels:
                labels = daily.index.astype(str).tolist()
            choice = st.selectbox("Pick a daily record to predict", labels)
            # extract index from label (last part after idx:)
            try:
                idx = int(choice.split("idx:")[-1])
            except Exception:
                idx = int(choice)

            row = daily.loc[idx]

            # Build feature vector
            X_row = pd.DataFrame(index=[0])
            # mapping for classification
            cls_map = {"Fear": 0.0, "Greed": 1.0, "Neutral": 0.5, "Unknown": 0.5}
            for fn in feature_names:
                if fn in daily.columns:
                    val = row.get(fn)
                    if fn == "classification":
                        val = cls_map.get(str(val), 0.5)
                    X_row.loc[0, fn] = val
                else:
                    # fill missing with median from daily
                    if fn in daily.columns:
                        X_row.loc[0, fn] = row.get(fn)
                    else:
                        X_row.loc[0, fn] = daily[fn].median() if fn in daily.columns else 0

            # Ensure numeric and fillna
            X_row = X_row.fillna(X_row.median(numeric_only=True)).astype(float)

            # Scale and predict
            X_scaled = scaler.transform(X_row)
            proba = model.predict_proba(X_scaled)[0]
            # Decision threshold control
            thresh = st.slider("Loss probability threshold", 0.0, 1.0, 0.5, 0.01)
            # apply threshold to determine predicted label (override model predict if desired)
            # assume 'loss' is one of the classes
            # determine class names
            try:
                classes = list(le.classes_)
            except Exception:
                classes = ["loss", "profit"]
            # map probabilities
            prob_map = {classes[i]: float(proba[i]) for i in range(len(classes))}
            # predicted class (standard) and threshold-adjusted
            pred_idx = int(model.predict(X_scaled)[0])
            pred_label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
            loss_prob = prob_map.get("loss", 0.0)
            # threshold decision
            pred_label_threshold = "loss" if loss_prob >= thresh else "profit"

            st.subheader("Prediction Result")
            st.write(f"Predicted class (model): **{pred_label}**")
            st.write(f"Predicted class (threshold {thresh:.2f}): **{pred_label_threshold}**")
            st.write(pd.DataFrame.from_dict(prob_map, orient="index", columns=["probability"]).sort_values("probability", ascending=False))

            # Recommendation
            loss_prob = prob_map.get("loss", 0.0)
            if loss_prob >= 0.5:
                st.warning(f"High predicted probability of loss ({loss_prob:.2f}). Consider risk-reducing actions:")
                st.markdown("- Reduce leverage and position size\n- Tighten stop-loss levels\n- Avoid adding to positions; consider hedging")
            else:
                st.success(f"Low predicted probability of loss ({loss_prob:.2f}). Consider measured scaling:")
                st.markdown("- Consider small scale-ins, keep risk per trade limited, monitor top risk features")

            # Show top feature importances and the values for this row
            feimp = saved.get("feature_importances") or {}
            if feimp:
                fi_df = pd.DataFrame.from_dict(feimp, orient="index", columns=["importance"]).sort_values("importance", ascending=False)
                st.subheader("Top features (global importance)")
                st.table(fi_df.head(10))
                # show values for top 5
                top_feats = fi_df.head(5).index.tolist()
                st.subheader("Selected record — top feature values")
                vals = {f: row.get(f) if f in row.index else None for f in top_feats}
                st.table(pd.Series(vals, name="value").to_frame())

            # Local explanation via SHAP (if available)
            try:
                import shap
            except Exception:
                st.caption('SHAP not installed — install `shap` to enable per-prediction explanations.')
            else:
                st.subheader("Local explanation (SHAP)")
                try:
                    # Try TreeExplainer first; fall back to generic Explainer using the selected row as background
                    explainer = None
                    try:
                        explainer = shap.TreeExplainer(model)
                    except Exception:
                        try:
                            explainer = shap.Explainer(model, X_row)
                        except Exception:
                            explainer = None

                    if explainer is not None:
                        shap_values = explainer.shap_values(X_row)
                        # shap_values format differs by model/library version. Normalize to 2-d array `sv` with shape (1, n_features)
                        sv = None
                        try:
                            import numpy as _np
                            # If list/tuple, try to select the class of interest
                            if isinstance(shap_values, (list, tuple)):
                                # prefer 'loss' class if available
                                try:
                                    pos_name = 'loss' if 'loss' in classes else classes[-1]
                                    pos_idx = classes.index(pos_name)
                                except Exception:
                                    pos_idx = 0
                                if pos_idx < len(shap_values):
                                    sv = _np.array(shap_values[pos_idx])
                                else:
                                    sv = _np.array(shap_values[0])
                            else:
                                arr = _np.array(shap_values)
                                if arr.ndim == 3:
                                    # common shapes: (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
                                    if arr.shape[1] == X_row.shape[1]:
                                        # shape (n_samples, n_features, n_classes)
                                        try:
                                            pos_name = 'loss' if 'loss' in classes else classes[-1]
                                            pos_idx = classes.index(pos_name)
                                        except Exception:
                                            pos_idx = arr.shape[2] - 1
                                        sv = arr[:, :, pos_idx]
                                    elif arr.shape[0] == len(classes):
                                        # shape (n_classes, n_samples, n_features)
                                        try:
                                            pos_name = 'loss' if 'loss' in classes else classes[-1]
                                            pos_idx = classes.index(pos_name)
                                        except Exception:
                                            pos_idx = 0
                                        sv = arr[pos_idx]
                                    else:
                                        # fallback: collapse to 2-d
                                        sv = arr.reshape(arr.shape[0], -1)
                                elif arr.ndim == 2:
                                    sv = arr
                                else:
                                    sv = arr.reshape(1, -1)

                            # ensure sv is 2-d with shape (1, n_features)
                            if sv is None:
                                raise ValueError('Could not normalize SHAP values to 2D')
                            if sv.ndim == 1:
                                sv = sv.reshape(1, -1)
                            # create DataFrame and display
                            sv_df = pd.DataFrame(sv, columns=X_row.columns)
                            st.table(sv_df.T.rename(columns={0: 'shap_value'}).sort_values('shap_value', ascending=False).head(10))
                            # plot bar for sv (use sv rather than raw shap_values)
                            try:
                                fig = shap.plots.bar(sv, max_display=10, show=False)
                                st.pyplot(fig)
                            except Exception:
                                pass
                        except Exception as _shap_err:
                            st.caption(f'SHAP processing failed: {_shap_err}')
                    else:
                        st.caption('SHAP explainer not available for this model type.')
                except Exception as e:
                    st.caption(f'SHAP explainer failed: {e}')

        except Exception as e:
            st.error(f"Error making prediction: {e}")

elif page == "Model Performance":
    st.header("📊 Model Performance - Binary Classification")
    st.markdown("""
    **Predictive Model**: Next-Day Profitability (Binary)
    - **Positive Class**: 'profit' (PnL ≥ 0)
    - **Negative Class**: 'loss' (PnL < 0)
    
    Binary classification provides clearer signals and typically yields better results than multi-class prediction.
    """)
    
    report_path = OUTPUTS / "classification_report.csv"
    cm_path = OUTPUTS / "confusion_matrix.csv"
    
    if report_path.exists():
        st.subheader("Classification Metrics")
        report = pd.read_csv(report_path, index_col=0)
        st.dataframe(report, use_container_width=True)
        
        # Display key metrics highlighted
        col1, col2, col3, col4 = st.columns(4)
        try:
            with col1:
                st.metric("Accuracy", f"{report.loc['accuracy', 'precision']:.3f}" if 'accuracy' in report.index else "N/A")
            with col2:
                st.metric("Precision", f"{report.loc['weighted avg', 'precision']:.3f}" if 'weighted avg' in report.index else "N/A")
            with col3:
                st.metric("Recall", f"{report.loc['weighted avg', 'recall']:.3f}" if 'weighted avg' in report.index else "N/A")
            with col4:
                st.metric("F1-Score", f"{report.loc['weighted avg', 'f1-score']:.3f}" if 'weighted avg' in report.index else "N/A")
        except:
            pass
    else:
        st.warning("Classification report not found. Run the pipeline to generate it.")
    
    if cm_path.exists():
        st.subheader("Confusion Matrix")
        cm = pd.read_csv(cm_path, index_col=0)
        st.dataframe(cm, use_container_width=True)
        
        # Visualize confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix - Binary Classification")
        st.pyplot(fig)
    else:
        st.warning("Confusion matrix not found. Run the pipeline to generate it.")

    # Feature importances (global interpretation)
    fi_path = OUTPUTS / "feature_importances.csv"
    if fi_path.exists():
        st.subheader("Feature Importances (global)")
        fi = pd.read_csv(fi_path, index_col=0)
        st.dataframe(fi.head(20), use_container_width=True)
        # bar chart
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            fi.sort_values("importance", ascending=True).tail(15).plot(kind="barh", legend=False, ax=ax)
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            ax.set_title("Top Features")
            st.pyplot(fig)
        except Exception:
            pass
