import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def generate_evaluation_charts():
    print("📊 Generating AI Evaluation Charts...")
    try:
        df = pd.read_csv("modern_heart_data.csv")
        model = joblib.load('modern_xgboost_model.pkl')
    except FileNotFoundError:
        print("❌ Error: Run main.py first to train the model.")
        return

    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Chart 1: Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted: Low Risk', 'Predicted: High Risk'], yticklabels=['Actual: Low Risk', 'Actual: High Risk'], annot_kws={"size": 16})
    plt.title(f'AI Diagnostic Accuracy\nOverall Accuracy: {accuracy:.2%}', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()

    # Chart 2: Feature Importance
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    feature_df['Feature'] = feature_df['Feature'].str.replace('_', ' ').str.title()
    sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis', hue='Feature', legend=False)
    plt.title('What Drives the AI\'s Decisions?', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Impact on Prediction', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()
    print("✅ Saved 'confusion_matrix.png' and 'feature_importance.png'")

if __name__ == "__main__":
    generate_evaluation_charts()