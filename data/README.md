# Input preparation

This code-only distribution contains no input datasets, experimental results or model weights.

## Data source: GATECDA

**The processed benchmark inputs used by RADVL were obtained from the public data release accompanying GATECDA (Deng et al., 2022).** These inputs include the circRNA–drug association matrix, host-gene sequence similarity matrix and drug structural similarity matrix.

- Original code and data repository: [GATECDA](https://github.com/yjslzx/GATECDA).
- Original publication: Deng, L., Liu, Z., Qian, Y., and Zhang, J. (2022). *Predicting circRNA-drug sensitivity associations via graph attention auto-encoder*. **BMC Bioinformatics, 23**, 160. [https://doi.org/10.1186/s12859-022-04694-y](https://doi.org/10.1186/s12859-022-04694-y).

The upstream GATECDA study derived circRNA–drug association statistics from CircRiC, host-gene sequences from NCBI Gene and drug structural information from PubChem. CircRiC is described by Ruan et al. (2019), *Comprehensive characterization of circular RNAs in ~1000 human cancer cell lines*, **Genome Medicine, 11**, 55. Please acknowledge GATECDA and the relevant original data providers when using these resources.

## Required input files

Provide CSV files with a header row and an index column:

| File | Reference shape | Meaning |
|---|---|---|
| `association.csv` | 271 × 218 | Binary selected association matrix; 4,134 positive entries in the reference benchmark |
| `gene_seq_sim.csv` | 271 × 271 | Fixed similarity of the corresponding host-gene sequences |
| `drug_str_sim.csv` | 218 × 218 | Fixed drug structural similarity |

The row and column labels of each similarity matrix must exactly follow the appropriate association-matrix axis. Additional mapping files may be supplied for interpreting generated predictions; label names alone do not resolve a specific circRNA BSJ.

Obtain the relevant processed matrices from the GATECDA data release and preserve the original identifier order when preparing the filenames above. Source association statistics alone do not reconstruct the fixed sequence and structure similarity matrices. CircRNA-side source labels are annotated by host genes, and the sequence similarity describes those host-gene sequences.

After placing these files here, run `python prepare_inputs.py`. Input preparation creates partition files locally under `protocols/`; these generated files are excluded from Git.
