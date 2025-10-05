# Final Comparison of Binary Classification Models
import pandas as pd

# Show final results with ensemble models
print("🏆 FINAL BINARY CLASSIFICATION RESULTS COMPARISON")
print("=" * 80)

# Create updated comparison with all models including ensemble
final_comparison = pd.DataFrame({
    model_name: {
        'F1_Score': results.get('f1', 0),
        'ROC_AUC': results.get('roc_auc', 0),
        'Accuracy': results.get('accuracy', 0),
        'Precision': results.get('precision', 0),
        'Recall': results.get('recall', 0)
    }
    for model_name, results in all_results.items()
}).T

final_comparison = final_comparison.round(4)
print(final_comparison)

# Find overall best model
best_roc_auc = final_comparison['ROC_AUC'].max()
best_model_by_roc = final_comparison['ROC_AUC'].idxmax()

best_f1 = final_comparison['F1_Score'].max()
best_model_by_f1 = final_comparison['F1_Score'].idxmax()

print(f"\n🥇 BEST PERFORMANCE SUMMARY:")
print(f"🎯 Best ROC-AUC: {best_model_by_roc} ({best_roc_auc:.4f})")
print(f"🎯 Best F1-Score: {best_model_by_f1} ({best_f1:.4f})")

# Model ranking
print(f"\n📊 MODEL RANKING BY ROC-AUC:")
ranking = final_comparison.sort_values('ROC_AUC', ascending=False)
for i, (model, row) in enumerate(ranking.iterrows(), 1):
    print(f"{i}. {model}: {row['ROC_AUC']:.4f} (F1: {row['F1_Score']:.4f})")