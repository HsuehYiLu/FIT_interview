import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# import packages for machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_curve, auc

def main():
    # 1. Load data
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
    except FileNotFoundError: # Handle the case where files are not found
        print("Error: 'train.csv' or 'test.csv' not found in the current directory.")
        return

    print("Data loaded successfully.")

    # Target variable
    target = 'Exited'
    
    # Identify columns to drop (identifiers that don't add predictive value)
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    drop_cols_train = [c for c in cols_to_drop if c in train_df.columns]
    drop_cols_test = [c for c in cols_to_drop if c in test_df.columns]

    X = train_df.drop(columns=[target] + drop_cols_train)
    y = train_df[target]
    
    X_test_final = test_df.drop(columns=drop_cols_test)

    # 2. Preprocessing Pipeline 
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 3. Model Pipeline Definition
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # 4. Train & Validation Split for Internal Evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training the model...")
    model.fit(X_train, y_train)

    # 5. Validation & Evaluation
    print("Evaluating model on validation set...")
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]

    # Calculate Metrics (基本的一些評估指標)
    f1 = f1_score(y_val, val_preds)
    cm = confusion_matrix(y_val, val_preds)
    report = classification_report(y_val, val_preds)
    fpr, tpr, thresholds = roc_curve(y_val, val_probs)
    roc_auc = auc(fpr, tpr)

    # Save Evaluation to text file (存一個檔案紀錄評估結果)
    with open('evaluation_metrics.txt', 'w') as f:
        f.write("Model Evaluation on 20% Validation Split\n")
        f.write("=========================================\n\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC-AUC Score: {roc_auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print("Saved evaluation metrics to 'evaluation_metrics.txt'.")

    # 6. Plotting (視覺化依些需要的圖表)
    sns.set_theme(style="whitegrid")

    # Plot 1: Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Validation Set)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plot_confusion_matrix.png')
    plt.close()

    # Plot 2: ROC Curve
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('plot_roc_curve.png')
    plt.close()

    # Plot 3: Feature Importance
    rf_classifier = model.named_steps['classifier']
    feature_names = numeric_features + \
                    list(model.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .get_feature_names_out(categorical_features))
    
    importances = rf_classifier.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(10) # Top 10

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('plot_feature_importance.png')
    plt.close()
    
    print("Saved visualizations (Confusion Matrix, ROC Curve, Feature Importances).")

    # 7. Final Prediction on Test Data
    print("Generating predictions for test.csv...")
    # Retrain on the entire training set for maximum data utilization
    model.fit(X, y)
    
    test_preds = model.predict(X_test_final)
    test_probs = model.predict_proba(X_test_final)[:, 1]

    # Create submission/prediction dataframe
    predictions_df = test_df.copy()
    predictions_df['Exited_Probability'] = test_probs
    predictions_df['Exited_Prediction'] = test_preds

    # Output to CSV
    predictions_df.to_csv('predictions.csv', index=False)
    print("Successfully saved predictions.")

if __name__ == "__main__":
    main()

