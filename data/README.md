# Data

The cohort data are **not included** in this repository, because they contain individual-level
participant information subject to privacy and data-governance restrictions.

To reproduce the analyses, place the file

    cMother_Chemical_data_25_3_2025.csv

in this folder (or set `PCB_DATA=/path/to/file.csv`). It is available from the data custodians
on reasonable request and subject to ethical approval.

Expected content: 102 rows; maternal covariates (`m_age`, `prepregancy_bmi`, `Maternal_education`,
`Maternal_smoking`, `Maternal_alcohol`, `Maternal_GDM`, `Maternal_BP`), lipids (`LDLC_m_mmol/L`,
`HDLC_m_mmol/L`, `TRIGL_m_mmol/L`, `CHOL_m_mmol/L`, `total lipid g/L_mother`) and 13 PCB congeners
in ng/L (columns `mPCB 28_ng/l` ... `mPCB 187_ng/l`). Values below the limit of detection are
stored as text such as `<2,46` (comma decimal separator).
