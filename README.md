# The Computational Quest for Precision in Leukemia Diagnosis

<details open>
    <summary><h2>🎯 Mission: Precision in Blood Cancer Diagnosis</h2></summary>

The term "__leukemia__" covers a spectrum of devastating blood cancers, but not all leukemias are the same. Misclassification leads to missed treatment windows and poor patient outcomes.

This project, developed for the __Programming for Bioinformatics__ course, tackles this critical challenge head-on. Using machine learning and public genomic data (gene expression profiles from the GSE13164 dataset) to build a robust, data-driven classifier.

The Goal: To accurately distinguish and predict the __four major leukemia subtypes__ based on their molecular signatures.

<table align="center">
  <thead>
    <tr>
      <th>Leukemia Sub-type</th>
      <th>Abbreviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Acute Lymphoblastic Leukemia</td>
      <td>ALL</td>
    </tr>
    <tr>
      <td>Acute Myeloid Leukemia</td>
      <td>AML</td>
    </tr>
    <tr>
      <td>Chronic Lymphoblastic Leukemia</td>
      <td>CLL</td>
    </tr>
   <tr>
      <td>Chronic Myeloid Leukemia</td>
      <td>CML</td>
    </tr>
  </tbody>
</table>

</details>

<details open>
<summary><h2>🛠️ The Pipeline: From Raw Data to Diagnostic Model</h2></summary>

This repository contains the complete pipeline for multi-class classification:

1. __Data Wrangling__ : Cleaning and preprocessing raw microarray data.
```text
    data_wrangling.py
    GSE13164_cleaned_features.csv
    GSE13164_cleaned_labels.csv
```

2. __Exploratory Data Analysis__ : Identifying key genes showing differential expression across subtypes.

3. __Feature Selection__ : Reducing the feature space to a high-impact biomarker panel.

4. __Model Development__ & Evaluation : Training and optimizing classifiers (e.g., Random Forest, Logistic Regression) using rigorous cross-validation to achieve high prediction accuracy.

</details>

<br>

<hr>

<p align="center">
Group 12 | SECB3203_25261 Programming for Bioinformatics
<p>
<p align="center">
**NGU YU LING** (A23CS0149)
</p>