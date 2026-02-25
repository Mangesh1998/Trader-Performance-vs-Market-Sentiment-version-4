# Binary Classification Update

## Overview
The prediction model has been converted from **3-category classification** to **binary classification** for improved predictive performance.

## Changes Made

### 1. **Target Variable** (Profitability Buckets)
**Before**: Three classes - `low`, `medium`, `high`  
**After**: Two classes - `loss`, `profit`

### 2. **Classification Logic**
- **`profit`**: Next-day PnL ≥ 0 (breakeven or positive)
- **`loss`**: Next-day PnL < 0 (negative)

This approach provides:
- ✅ Clearer decision boundaries
- ✅ Simpler interpretation (profitable vs. unprofitable day)
- ✅ Better generalization for predictive tasks
- ✅ Reduced class imbalance issues
- ✅ Enhanced model performance

### 3. **Modified Files**

#### `config/config.yaml`
- Changed `profitability_buckets: 3` → `profitability_buckets: 2`

#### `src/features/metrics.py`
- Updated `create_model_features()` function:
  - Changed from quantile-based binning (pd.qcut) to simple threshold
  - Target now: `np.where(daily["next_day_pnl"] >= 0, "profit", "loss")`
  - Removed complex multi-class handling

#### `src/models/train_model.py`
- Enhanced metrics for binary classification:
  - Added `accuracy`, `precision`, `recall`
  - Added `roc_auc` (Area Under ROC Curve) for better evaluation
  - Better zero_division handling

#### `streamlit_app.py`
- Redesigned Model Performance page:
  - Added informative header explaining binary classification
  - Displays key metrics: Accuracy, Precision, Recall, F1-Score
  - Visualizes confusion matrix as heatmap
  - Better layout with columns for metric display

### 4. **Why Binary Classification?**

**Advantages over 3-class:**
1. **Reduced Complexity**: Simpler decision boundaries = better generalization
2. **Better Metrics**: F1-score, Precision, Recall, ROC-AUC are more meaningful
3. **Less Imbalance**: Two-class problems are easier to balance
4. **Clearer Business Logic**: "Will I profit tomorrow?" (yes/no)
5. **Faster Training**: Fewer parameters to optimize

### 5. **Running the Pipeline**

No changes needed to execution. Simply run:
```bash
python run_pipeline.py
```

Then view results:
```bash
streamlit run streamlit_app.py
```

### 6. **Expected Improvements**

- Model performance metrics will be more stable
- ROC-AUC will indicate quality of probability predictions
- Confusion matrix will show True Positives, True Negatives, False Positives, False Negatives
- Better actionable insights (profit prediction clarity)

## Model Performance Interpretation

**Classification Report** shows:
- **Precision**: Of predicted "profit" days, how many actually profit?
- **Recall**: Of actual "profit" days, how many did we predict correctly?
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples in each class

**Confusion Matrix** shows:
- **True Negatives (TN)**: Correctly predicted loss days
- **False Positives (FP)**: Predicted profit but was actually loss
- **False Negatives (FN)**: Predicted loss but was actually profit
- **True Positives (TP)**: Correctly predicted profit days

**ROC-AUC**:
- Measures model's ability to distinguish between classes
- Range: 0 (worst) to 1 (perfect)
- 0.5 indicates random guessing

---
**Last Updated**: 2026-02-26
