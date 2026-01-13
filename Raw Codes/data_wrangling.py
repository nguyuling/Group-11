import GEOparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re 
import os 
from functools import reduce

# configuration
GSE_ID = "GSE13164" 
OUTPUT_CLEAN_DATA = "GSE13164_cleaned_features.csv"
OUTPUT_CLEAN_LABELS = "GSE13164_cleaned_labels.csv"

# 4 target classes of leukemia
TARGET_CLASSES = ['ALL', 'AML', 'CLL', 'CML']
FEATURE_IDENTIFIER = 'GB_ACC'

#! 1. GEO Download, Parse, and Extraction
def load_and_preprocess_geo(gse_id):
    
    print(f"Downloading and parsing GEO series: {gse_id}. This may take a few minutes...")
    try:
        # downloads file and parses it
        gse = GEOparse.get_GEO(geo=gse_id, destdir="./", silent=False)
    except Exception as e:
        print(f"Error downloading/parsing {gse_id}: {e}")
        return None, None

    platform_key = list(gse.gpls.keys())[0]
    gpl_table = gse.gpls[platform_key].table.copy()
    
    #! 2. Data Wrangling: label extraction and filtering
    print("\nStarting Label Extraction and Filtering...")
    
    metadata_list = []
    sample_dfs = []

    for name, gsm in gse.gsms.items():
        characteristics = ' '.join(gsm.metadata.get('characteristics_ch1', ['']))
        leukemia_type = None
        
        # identify the leukemia type based on target classes
        for t_type in TARGET_CLASSES:
            if re.search(r'\b' + re.escape(t_type) + r'\b', characteristics, re.IGNORECASE):
                leukemia_type = t_type
                break

        # only process samples belonging to any target class
        if leukemia_type in TARGET_CLASSES:
            # add sample info to metadata list
            metadata_list.append({'Sample_ID': name, 'Leukemia_Type': leukemia_type})
            
            # extract expression data from the GSM's table
            gsm_df = gsm.table.copy()
            
            # use 'ID_REF' as the joining key (probe ID)
            if 'ID_REF' not in gsm_df.columns:
                print(f"Warning: GSM {name} missing 'ID_REF'. Skipping.")
                continue
            
            # identify the expression value column (2000+)
            value_col = next((col for col in gsm_df.columns if col.upper() in ['VALUE', 'LOG_RATIO', 'SIGNAL', 'AVG_SIGNAL', 'NORMALIZED_SIGNAL']), None)
            
            # fallback
            if not value_col:
                non_id_cols = [c for c in gsm_df.columns if c != 'ID_REF']
                if non_id_cols:
                    value_col = non_id_cols[-1]
                else:
                    print(f"Warning: GSM {name} has no detectable value column. Skipping.")
                    continue

            # select ID_REF and the value, then rename the value column to the sample name
            gsm_df = gsm_df[['ID_REF', value_col]].rename(columns={value_col: name})
            sample_dfs.append(gsm_df)

    metadata_df = pd.DataFrame(metadata_list)
    print(f"Samples identified and labeled: {len(sample_dfs)}.")
    
    if not sample_dfs:
        print("CRITICAL ERROR: No samples with identifiable labels or expression values were found.")
        return None, None

    #! 3. Merging Sample DataFrames
    print("\nMerging individual sample expression tables...")
    
    # merge all sample DataFrames based on ID_REF
    expression_data = reduce(lambda left, right: pd.merge(left, right, on='ID_REF', how='inner'), sample_dfs)
    expression_data = expression_data.set_index('ID_REF')
    
    print(f"Initial Merged Feature Matrix Shape (Probes x Samples): {expression_data.shape}")

    #! 4. Probe Mapping and Aggregation
    print("\nStarting Probe Mapping and Aggregation using GPL table...")
    
    # get the necessary columns from the GPL annotation table
    annotation_cols = ['ID', FEATURE_IDENTIFIER]
    annotation_df = gpl_table[annotation_cols].rename(columns={'ID': 'ID_REF', FEATURE_IDENTIFIER: 'Feature_ID'})

    # clean feature IDs (GB_ACC)
    annotation_df.dropna(subset=['Feature_ID'], inplace=True)
    annotation_df = annotation_df[annotation_df['Feature_ID'].str.strip() != '---']
    annotation_df['Feature_ID'] = annotation_df['Feature_ID'].apply(lambda x: x.split(' // ')[0].strip())
    
    # merge annotation with expression data
    merged_df = pd.merge(expression_data.reset_index(), annotation_df, on='ID_REF', how='inner')

    # remove probes that didn't map to a valid feature_ID
    merged_df.dropna(subset=['Feature_ID'], inplace=True)
    
    # get the list of sample columns (all columns except ID_REF and Feature_ID)
    sample_cols = [col for col in merged_df.columns if col.startswith('GSM')]

    # aggregate / group by feature ID (GB_ACC) and the mean expression
    final_features_df = merged_df.groupby('Feature_ID')[sample_cols].mean()
    
    # transpose the final data (samples as rows, features/accessions as cols)
    final_features_df = final_features_df.T
    
    #! 5. Final Data Alignment and Encoding

    # Make sure the metadata index aligns with the sample IDs found in the feature
    metadata_df = metadata_df.set_index('Sample_ID').loc[final_features_df.index.tolist()].reset_index()

    # encode the target labels
    le = LabelEncoder()
    metadata_df['Target_Code'] = le.fit_transform(metadata_df['Leukemia_Type'])

    # final check
    print("\n--- Final Data Alignment ---")
    print(f"Final Feature Matrix Shape (Samples x Features): {final_features_df.shape}")
    print(f"Final Label Matrix Shape (Samples x Info): {metadata_df.shape}")
    print(f"Sample-wise alignment check (should be TRUE): {all(final_features_df.index == metadata_df['Sample_ID'])}")

    return final_features_df, metadata_df

#! Execution
# create data directory
os.makedirs('data', exist_ok=True)

print("--- Starting O1: Data Collection & Wrangling (Python) ---")
# load, clean, and map the data
expression_data, metadata = load_and_preprocess_geo(GSE_ID)

if expression_data is not None and metadata is not None:
    # save the final products for next step
    try:
        expression_data.to_csv(OUTPUT_CLEAN_DATA)
        metadata.to_csv(OUTPUT_CLEAN_LABELS, index=False)
        print(f"\nSUCCESS: Cleaned features saved to: {OUTPUT_CLEAN_DATA}")
        print(f"SUCCESS: Cleaned labels saved to: {OUTPUT_CLEAN_LABELS}")
    except Exception as e:
        print(f"Error saving data: {e}")

    print("\nData Wrangling (O1) is now complete. You can proceed to EDA and Feature Selection (O2/O3) in 'leukemia_eda_feature_selection.py'.")