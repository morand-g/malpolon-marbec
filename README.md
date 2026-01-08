## Malpolon - minimal version

Use case: regression or classification on many species based on multimodal input

### Installation

Add the package to your environment with this command:

```
pip install -e ./malpolon
```

### Usage

Most parameters can be chosen from the yaml config files, without editing the python code.
For instance, to pretrain a foundation model, set *data.mask_inputs* to a value greater than 0 (and smaller than 1).

Four sample config files are provided:
- *fm* for the foundation model,
- *reg* for biomass regression
- *binned* for biomass classes
- *binned_pa* for Presence/Absence

To train a model, set *run.predict* to *false* and chose your hyperparameters.