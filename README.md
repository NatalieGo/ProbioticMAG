# ProbioticMAG
Interpretable machine learning framework for probiotic potential assessment of metagenome-assembled genomes


--------------------
Dataset labels were obtained from the iProbiotics database.

Reference:
Zhang et al., iProbiotics: a machine learning platform for probiotic genome prediction.

Original resource: [iprobiotics](https://iprobiotics.enfc.cn/)


--------------------
Bakta was used to annotate reference genomes and reconstructed MAGs. Functional annotations were parsed to identify genes related to adhesion, stress response, antimicrobial activity, bile salt metabolism, biosafety (AMR/virulence), and phage defense systems. The detected genes were aggregated into biologically interpretable functional modules that formed the feature space for machine learning models.

Software: [Bakta](https://github.com/oschwengers/bakta)
Publication: [Schwengers et al., 2021 (Microbial Genomics)](https://doi.org/10.1099/mgen.0.000685)
