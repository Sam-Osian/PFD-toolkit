# Descriptions of files

## /data
* `all_tagged_reports.csv` - the entire corpus of PFD reports from July 2013 to November 2023, tagged by whether PFD Toolkit determined whether it reflected a child-suicide case or not.
* `child_suicide_cases.csv` - as above, but only child suicide cases. Reports were coded by PFD Toolkit in line with the ONS analytical frame.
* `dataset_for_annotation.csv` - all reports that PFD Toolkit identified as being a child-suicide case, with 73 randomly sampled decoy cases. 


## /scripts
* `fieldwork.ipynb` - the entire fieldwork, performed by PFD Toolkit. The "5 minutes, 29 seconds" runtime quoted in the manuscript is evidenced in this notebook.
* `get_decoys.ipynb` - the script for randomly sampling 73 decoy cases and adding it to the 73 child-suicide cases identified by PFD Toolkit.
* `kappa_results.R` - the script for analysing post-consensus and PFD Toolkit annotations and producing kappa figures.
* `power_analysis.R` - the script for conducting the power analysis for this research.

## Other files
* `PFD Toolkit--Consensus Comparison.xlsx` - a case-level record of how each report in the 146 sample was assigned by (i) PFD Toolkit; (ii) Each individual clinical annotator; (iii) The post-consensus annotation activity. 

