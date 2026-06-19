# ProbioticMAG

Interpretable machine learning framework for probiotic potential assessment of metagenome-assembled genomes (MAGs).

## Overview

ProbioticMAG is a machine learning-based framework developed as part of a Master's thesis project. The workflow combines genome annotation, biologically interpretable feature engineering, and machine learning to assess the potential probiotic status of metagenome-assembled genomes (MAGs).

The framework was applied to MAGs reconstructed from the metagenome of a traditional farmer cheese and provides a computational ranking of candidate probiotic strains.

## Input Features

Features are aggregated into biologically interpretable functional modules:

* Adhesion
* Stress response
* Antimicrobial activity
* Bile salt metabolism
* Phage defense systems
* Biosafety-related factors (AMR and virulence)

## Machine Learning Models

The following models were evaluated:

* Random Forest (scikit-learn)
* XGBoost

Model interpretation was performed using SHAP (SHapley Additive exPlanations).

## Repository Structure

* `data/` – datasets and feature matrices
* `notebooks/` – model training and analysis notebooks
* `results/` – model outputs, figures, and rankings
* `docs/` – feature descriptions and references

## Data Sources

Dataset labels were obtained from the iProbiotics database.

Reference:
Zhang et al. *iProbiotics: a machine learning platform for probiotic genome prediction.*

Original resource: [iprobiotics](https://iprobiotics.enfc.cn/)

## Genome Annotation

Bakta was used to annotate reference genomes and reconstructed MAGs. Functional annotations were parsed to identify genes related to adhesion, stress response, antimicrobial activity, bile salt metabolism, biosafety (AMR/virulence), and phage defense systems. The detected genes were aggregated into biologically interpretable functional modules that formed the feature space for machine learning models.

Software: [Bakta](https://github.com/oschwengers/bakta)

Publication:
[Schwengers O. et al. (2021). Bakta: rapid and standardized annotation of bacterial genomes via alignment-free sequence identification. Microbial Genomics.](https://doi.org/10.1099/mgen.0.000685)

## Citation

If you use this repository, please cite the original software tools and databases referenced above.
