# aqua-py-analysis

Analsysis codebase for paper:
*[ Ca+ activity maps of astrocytes tagged by axoastrocytic AAV transfer ](https://www.science.org/doi/10.1126/sciadv.abe5371)*

## Requirements:
- Python 3.7

Additional requirements for AquA preprocessing:
- Matlab 2019a+

## Datasets

| Dataset available **[here](https://www.dropbox.com/sh/csaxn3o84zchh4g/AABZge806LiT7rinoWtOnXMYa?dl=0)** | |
| ------------- | ------------- |
| [Dataset structure](https://www.dropbox.com/s/csgi2j146sfxjsu/datasets_structure.txt?dl=0) | Dataset file structure |
| [First astrocyte experiment dataset](https://www.dropbox.com/s/9n7aufr6fuuo101/astro_first_experiment.zip?dl=0)  | **Main dataset.** Used to assess spatiotemporal organisation of astrocyte calcium signals during different behavioural states  - *rest, run, vibrissa stimulation, vibrissa exploration,* and to generate the activity heatmaps|
| [Second astrocyte experiment dataset](https://www.dropbox.com/s/95b29bm7avrhn7z/astro_second_experiment.zip?dl=0) | Used to assess if multiple vibrissa stimuli alter calcium activity in astrocytes in relation to single vibrissae stimuli/no stimuli) |
| [Astro-axon dataset](https://www.dropbox.com/s/bcniyf365yl6sbd/astro_axons.zip?dl=0) | Axon and astrocyte calcium activity during behavioural and vibrissa stimulation |
| [Data plots curation SA](https://www.dropbox.com/sh/j43kndms5u8075g/AAApf4jrTZ7up2spS4OD0EFAa?dl=0) | Plots as presented in paper sorted by Figure |
| [Data plots](https://www.dropbox.com/sh/60cxgwx1s63ebdd/AAD698Fq2j_QIRV4Heox2C8Oa?dl=0) | All relevant generated plots |


## Loading datasets
| Notebooks  |                   |
| ------------- | ------------- |
| **[AstrocyteExample.ipynb](https://github.com/Achilleas/aqua-py-analysis/blob/master/AstrocyteExample.ipynb)**   | Astrocyte notebook example of dataset loading |
| **[AxonAstrocyteExample.ipynb](https://github.com/Achilleas/aqua-py-analysis/blob/master/AxonAstrocyteExample.ipynb)** | Axon-Astro notebook example of dataset loading |
| **Paper related notebooks** |          | 
| **[generate_astrocyte_paper_plots.ipynb](https://github.com/Achilleas/aqua-py-analysis/blob/master/generate_astrocyte_paper_plots.ipynb)**| Generates relevant paper plots from first/second astrocyte experiment dataset |
| **[generate_axon_paper_plots.ipynb](https://github.com/Achilleas/aqua-py-analysis/blob/master/generate_axon_paper_plots.ipynb)**| Generates relevant paper plots from astro-axon experiment dataset |
| **[generate_misc_paper_plots.ipynb](https://github.com/Achilleas/aqua-py-analysis/blob/master/generate_misc_paper_plots.ipynb)**| Generates  other remaining paper plots (capsid density plot, astrocyte density plot in different cortical layers, neuron density plot in different cortical layers) | 
| **[statistical_analysis.ipynb](https://github.com/Achilleas/aqua-py-analysis/blob/master/statistical_analysis.ipynb)**| Statistical analysis of datasets generated relevant to paper | 


## Preprocessing (creating your own dataset)
See **[preprocessing-README](https://github.com/Achilleas/aqua-py-analysis/tree/master/preprocessing/README.md)** for details

#### Important note
**The objective of this github repository is to replicate the analysis of AquA processed data generated for this paper. Although the codebase can be used for new experiments, it might have to be modified to account for different experimental conditions.**
