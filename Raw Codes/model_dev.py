import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
FEATURES_PATH = "GSE13164_cleaned_features.csv"
LABELS_PATH = "GSE13164_cleaned_labels.csv"
TOP_VAR_FEATURES_FOR_POLY = 30  # limit size for polynomial regression
RANDOM_STATE = 42


def load_and_prepare():
    features_df = pd.read_csv(FEATURES_PATH, index_col=0)
    labels_df = pd.read_csv(LABELS_PATH)

    # Align to shared samples
    combined = labels_df.set_index("Sample_ID").join(features_df, how="inner")

    # Choose target: prefer existing numeric Target_Code; else encode Leukemia_Type
    if "Target_Code" in combined.columns:
        y = combined["Target_Code"].astype(float)
    else:
        # Simple encoding fallback
        y = combined["Leukemia_Type"].astype("category").cat.codes.astype(float)

    # Drop non-feature columns for X
    meta_cols = [c for c in ["Leukemia_Type", "Target_Code"] if c in combined.columns]
    X = combined.drop(columns=meta_cols)

    return X, y, combined


def analyze_differential_genes(combined, X, top_n=20):
    """
    Identify genes that differ most between leukemia types.
    Uses ANOVA F-statistic for multi-class comparison.
    """
    print("\n" + "="*60)
    print("DIFFERENTIAL GENE EXPRESSION ANALYSIS")
    print("="*60)
    
    # Get leukemia type labels
    if "Leukemia_Type" in combined.columns:
        groups = combined["Leukemia_Type"]
    else:
        print("Warning: No Leukemia_Type column found")
        return None
    
    unique_types = groups.unique()
    print(f"\nComparing {len(unique_types)} leukemia types: {list(unique_types)}")
    print(f"Sample counts per type:")
    print(groups.value_counts())
    
    # Calculate F-statistic and p-value for each gene
    gene_stats = []
    for gene in X.columns:
        gene_expr = X[gene]
        # Group gene expression by leukemia type
        groups_data = [gene_expr[groups == lt].values for lt in unique_types]
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups_data)
        
        # Calculate mean expression per group
        group_means = {lt: gene_expr[groups == lt].mean() for lt in unique_types}
        
        gene_stats.append({
            'gene': gene,
            'f_statistic': f_stat,
            'p_value': p_value,
            **{f'mean_{lt}': group_means[lt] for lt in unique_types}
        })
    
    # Create DataFrame and sort by F-statistic
    diff_genes_df = pd.DataFrame(gene_stats)
    diff_genes_df = diff_genes_df.sort_values('f_statistic', ascending=False)
    
    # Display top differentially expressed genes
    print(f"\nTop {top_n} Differentially Expressed Genes:")
    print("-" * 60)
    
    display_cols = ['gene', 'f_statistic', 'p_value'] + [c for c in diff_genes_df.columns if c.startswith('mean_')]
    top_genes_df = diff_genes_df[display_cols].head(top_n)
    
    for idx, row in top_genes_df.iterrows():
        print(f"\n{row['gene']}")
        print(f"  F-statistic: {row['f_statistic']:.2f}, p-value: {row['p_value']:.2e}")
        for col in display_cols[3:]:
            leuk_type = col.replace('mean_', '')
            print(f"  {leuk_type}: {row[col]:.3f}")
    
    return diff_genes_df


def plot_top_genes(combined, X, diff_genes_df, top_n=5):
    """
    Visualize expression of top differentially expressed genes across leukemia types.
    """
    if diff_genes_df is None:
        return
    
    top_genes = diff_genes_df.head(top_n)['gene'].tolist()
    groups = combined["Leukemia_Type"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, gene in enumerate(top_genes[:6]):
        ax = axes[i]
        gene_expr = X[gene]
        
        # Create dataframe for plotting
        plot_data = pd.DataFrame({
            'Expression': gene_expr.values,
            'Leukemia_Type': groups.values
        })
        
        # Box plot
        sns.boxplot(data=plot_data, x='Leukemia_Type', y='Expression', ax=ax)
        ax.set_title(f'{gene}', fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        
    # Remove empty subplots
    for i in range(len(top_genes), 6):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.suptitle('Top Differentially Expressed Genes', y=1.02, fontsize=14, fontweight='bold')
    plt.show()
    
    # Create heatmap of top genes
    top_genes_for_heatmap = diff_genes_df.head(20)['gene'].tolist()
    heatmap_data = X[top_genes_for_heatmap].T
    
    # Sort samples by leukemia type for better visualization
    sorted_indices = np.argsort(groups.values)
    heatmap_data = heatmap_data.iloc[:, sorted_indices]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, 
                yticklabels=True, xticklabels=False,
                cbar_kws={'label': 'Expression Level'})
    plt.title('Top 20 Differentially Expressed Genes Heatmap', fontsize=14, fontweight='bold')
    plt.ylabel('Gene')
    plt.xlabel('Samples (sorted by Leukemia Type)')
    plt.tight_layout()
    plt.show()


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    print(f"\n{name}")
    print("-" * 60)
    print(f"Train R^2: {train_r2:.4f} | Test R^2: {test_r2:.4f}")
    print(f"Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")

    return {
        "name": name,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "y_test": y_test,
        "test_pred": test_pred,
    }


def plot_predictions(result):
    y_test = result["y_test"]
    test_pred = result["test_pred"]

    plt.figure(figsize=(10, 4))

    # Predicted vs True
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, test_pred, alpha=0.6)
    min_y, max_y = y_test.min(), y_test.max()
    plt.plot([min_y, max_y], [min_y, max_y], "r--", label="Ideal")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{result['name']}: True vs Pred")
    plt.legend()

    # Residuals
    plt.subplot(1, 2, 2)
    residuals = test_pred - y_test
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(f"{result['name']}: Residuals")

    plt.tight_layout()
    plt.show()


def main():
    X, y, combined = load_and_prepare()
    print(f"Loaded aligned data: X shape {X.shape}, y length {len(y)}")
    
    # ----------------------------------------------------------------
    # DIFFERENTIAL GENE EXPRESSION ANALYSIS
    # ----------------------------------------------------------------
    diff_genes_df = analyze_differential_genes(combined, X, top_n=20)
    
    if diff_genes_df is not None:
        # Visualize top differentially expressed genes
        plot_top_genes(combined, X, diff_genes_df, top_n=6)
        
        # Save results to CSV
        output_file = "differential_genes_analysis.csv"
        diff_genes_df.to_csv(output_file, index=False)
        print(f"\nDifferential gene analysis saved to: {output_file}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = []

    # ----------------------------------------------------------------
    # 1) Simple Linear Regression (single feature)
    # ----------------------------------------------------------------
    first_gene = X.columns[0]
    simple_model = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression()),
    ])
    res_simple = evaluate_model(
        f"Simple Linear Regression on {first_gene}",
        simple_model,
        X_train[[first_gene]],
        y_train,
        X_test[[first_gene]],
        y_test,
    )
    results.append(res_simple)

    # ----------------------------------------------------------------
    # 2) Multiple Linear Regression (all genes, scaled)
    # ----------------------------------------------------------------
    multi_model = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression()),
    ])
    res_multi = evaluate_model(
        "Multiple Linear Regression (all genes)",
        multi_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    results.append(res_multi)

    # ----------------------------------------------------------------
    # 3) Polynomial Regression (top-variance genes to keep it tractable)
    # ----------------------------------------------------------------
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(TOP_VAR_FEATURES_FOR_POLY).index.tolist()
    poly_model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("linreg", LinearRegression()),
    ])
    res_poly = evaluate_model(
        f"Polynomial Regression (degree=2, top {len(top_features)} genes)",
        poly_model,
        X_train[top_features],
        y_train,
        X_test[top_features],
        y_test,
    )
    results.append(res_poly)

    # Show metrics summary
    print("\nSummary of models:")
    for r in results:
        print(
            f"{r['name']}: Test R^2={r['test_r2']:.4f}, Test MSE={r['test_mse']:.4f}"
        )

    # Visualize predictions for the best test R^2 model
    best = max(results, key=lambda r: r["test_r2"])
    print(f"\nBest model by Test R^2: {best['name']}")
    plot_predictions(best)

    # Decision making / prediction example
    sample_preds = best["test_pred"][:5]
    print("\nSample predictions (first 5 of test set):")
    for i, pred in enumerate(sample_preds, 1):
        print(f"Sample {i}: predicted target {pred:.3f}")


if __name__ == "__main__":
    main()