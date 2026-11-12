# Input preparation

This code-only distribution contains no input datasets, experimental results or model weights.

Provide CSV files with a header row and an index column:

| File | Reference shape | Meaning |
|---|---|---|
| `association.csv` | 271 × 218 | Binary selected association matrix; 4,134 positive entries in the reference benchmark |
| `gene_seq_sim.csv` | 271 × 271 | Fixed similarity of the corresponding host-gene sequences |
| `drug_str_sim.csv` | 218 × 218 | Fixed drug structural similarity |

The row and column labels of each similarity matrix must exactly follow the appropriate association-matrix axis. Additional mapping files may be supplied for interpreting generated predictions; label names alone do not resolve a specific circRNA BSJ.

The study uses the GATECDA benchmark derived from CircRiC. Refer to Deng et al. (2022), *Predicting circRNA-drug sensitivity associations via graph attention auto-encoder*, BMC Bioinformatics 23:160; and Ruan et al. (2019), *Comprehensive characterization of circular RNAs in ~1000 human cancer cell lines*, Genome Medicine 11:55. Obtain the relevant processed matrices from the original benchmark provider and preserve the original identifier order. Source statistics alone do not reconstruct the fixed sequence and structure matrices.

After placing these files here, run `python prepare_inputs.py`. Input preparation creates partition files locally under `protocols/`; these generated files are excluded from Git.
