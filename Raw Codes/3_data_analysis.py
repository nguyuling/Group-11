import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# load the 2 cleaned CSVs
features_df = pd.read_csv('GSE13164_cleaned_features.csv', index_col=0)
labels_df = pd.read_csv('GSE13164_cleaned_labels.csv')
# Align and combine: Sample_ID | Leukemia_Type | Gene1 ... GeneN
combined_df = (
    labels_df[['Sample_ID', 'Leukemia_Type']]
    .set_index('Sample_ID')
    .join(features_df, how='inner')
    .reset_index()
)
combined_df = combined_df[['Sample_ID', 'Leukemia_Type'] + [c for c in combined_df.columns if c not in ['Sample_ID', 'Leukemia_Type']]]

#! 0. BASIC INFO
print("\n" + "-" * 60 + "\n0. BASIC INFO\n" + "-" * 60)
print(f"\nShape: {combined_df.shape}")
print(f"Columns: Sample_ID, Leukemia_Type, {combined_df.shape[1]-2} genes")
print(f"\nFirst 3 rows: \n{combined_df.iloc[:3, :5]}")
print(f"\nLast 3 rows: \n{combined_df.iloc[-3:, :-1]}")

# Separate features from metadata
gene_expression = combined_df.iloc[:, 2:]  # All columns after Sample_ID and Leukemia_Type
leukemia_types = combined_df['Leukemia_Type']


#! 1. DESCRIPTIVE STATISTICS
print("\n" + "-" * 60 + "\n1. DESCRIPTIVE STATISTICS\n" + "-" * 60)

# shape (x, y) of data
print(f"\nOverall gene expression statistics: \n{gene_expression.describe()}")

# First and last 3 rows of data
print("\nGene expression statistics by leukemia type:")
for leukemia_type in sorted(leukemia_types.unique()):
    mask = leukemia_types == leukemia_type
    print(f"\n{leukemia_type} (n={mask.sum()}):")
    print(gene_expression[mask].describe().loc[['mean', 'std', 'min', 'max']])


#! 2. BASIC GROUPING
print("\n" + "-" * 60 + "\n2. BASIC GROUPING ANALYSIS\n" + "-" * 60)

# mean of gene expression
grouped_stats = combined_df.groupby('Leukemia_Type')[gene_expression.columns].mean()
print(f"\nMean gene expression by leukemia type (first 10 genes): \n{grouped_stats.iloc[:, :10]}")

# leukemia type sample count
sample_counts = combined_df.groupby('Leukemia_Type').size()
print(f"\nSample counts by leukemia type: \n{sample_counts}")


#! 3. ANOVA (Analysis of Variance)
print("\n" + "-" * 60 + "\n3. ANOVA TEST\n" + "-" * 60)
print("Testing if gene expression differs significantly by leukemia type...\n")

# SciPy returns per-feature F and p when arrays are passed; summarize for readability
groups = [gene_expression[leukemia_types == ltype].values for ltype in sorted(leukemia_types.unique())]
anova_results = stats.f_oneway(*groups)

# summarize the distribution of F and p across genes
overall_f_mean = float(np.nanmean(anova_results.statistic))
overall_p_mean = float(np.nanmean(anova_results.pvalue))
overall_p_min = float(np.nanmin(anova_results.pvalue))

print(f"Mean F-statistic across genes: {overall_f_mean:.4f}")
print(f"Mean P-value across genes: {overall_p_mean:.4e}")
print(f"Minimum P-value across genes: {overall_p_min:.4e}")

if overall_p_min < 0.05:
    print("Result: At least one gene shows significant differences across leukemia types (min p < 0.05)")
else:
    print("Result: No genes show significant differences across leukemia types (min p >= 0.05)")

# per-gene ANOVA
print("\nPer-gene ANOVA results (first 10 genes):")
f_stats = []
p_values = []

for gene in gene_expression.columns:
    groups = [gene_expression.loc[leukemia_types == ltype, gene].values 
              for ltype in sorted(leukemia_types.unique())]
    f_stat, p_val = stats.f_oneway(*groups)
    f_stats.append(f_stat)
    p_values.append(p_val)

anova_df = pd.DataFrame({
    'Gene': gene_expression.columns,
    'F-Statistic': f_stats,
    'P-Value': p_values
}).sort_values('P-Value')

print(anova_df.head(10))


#! 4. CORRELATION ANALYSIS
print("\n" + "-" * 60 + "\n4. CORRELATION ANALYSIS\n" + "-" * 60)

# compute correlation matrix
correlation_matrix = gene_expression.corr()
print(f"\nCorrelation matrix shape: {correlation_matrix.shape}")
print("\nTop 5 gene pairs with highest correlation:")

# Get upper triangle to avoid duplicates
corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_pairs.append({
            'Gene1': correlation_matrix.columns[i],
            'Gene2': correlation_matrix.columns[j],
            'Correlation': correlation_matrix.iloc[i, j]
        })

corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
print(f"\nTop 5 gene pairs with highest correlation: \n{corr_pairs_df.head(5)}")
print(f"\nTop 5 gene pairs with lowest correlation: \n{corr_pairs_df.tail(5)}")