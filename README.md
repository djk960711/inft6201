# INFT6201: Modelling codebase
## Entry Point for Jupyter:
Run the code `run_pipeline.ipynb`. For geographical analysis, see `ChoropletMaps_GroupProject.ipynb`.
## Structure
The code was designed as a pipeline to perform all analysis with the following modularised python structure.

| **Path** | **Purpose** |
| --- | --- |
| _main.py_ | The entry point of the application. The flow of the pipeline is clear from this file. |
| _conf.yaml_ | This contains all key parameters used in the analysis, such as Thresholds, Regular expression conditions (used to extract value from the free text fields), Clustering parameters and Selected features. This allowed the analysis to be fine-tuned without having to alter individual functions. |
| _src/_ | This contained functions specific to elements of the pipeline in a clean structure. |
|
 | src/ingestion.py | This contained functions to dedupe and parse datetime columns. |
|
 | src/descriptive.py | This contained functions to provide descriptive statistics and plots of the data. |
|
 | src/feature.py | This contained functions to engineer new features, including clustering analysis. |
|
 | src/modelling.py | This contained functions to impute missing value, perform the logistic regression process and output diagnostics. |

The code is compliant with PEP-8 formatting specification for python code. The code required the following packages:

- imblearn: For applying SMOTE
- matplotlib: For plotting
- numpy: For basic manipulation of the data
- pandas: For basic manipulation of the data
- pyyaml: For reading a yaml-based config file
- scipy: For performing statistical analysis
- sklearn: For performing logistic regression.

Docstrings and comments have been placed in-line to provide context on the functionality of each component.