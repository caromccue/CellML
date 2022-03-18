# CellML

[![PyPI version](https://badge.fury.io/py/crystalml.svg)](https://badge.fury.io/py/crystalml)
[![Travis CI status](https://travis-ci.com/hlgirard/CrystalML.svg?branch=master)](https://travis-ci.com/hlgirard/CrystalML/branches)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/hlgirard/CrystalML/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/167236947.svg)](https://zenodo.org/badge/latestdoi/167236947)

**Disclaimer:** This program is undergoing active development and should _not_ be used for production. All APIs and commands are susceptible to change without notice.

Integrated tool to segment individual cells from optical cell images and use machine learning to determine whether cells are adherent or not based on their shapes.

From a directory containing a time-series of images of cells on surfaces, the tool segments individual cells and uses a pre-trained CNN model to determine whether each cell is adherent or not.

## Installation

### Install with pip

CellML is on PyPI so it can be installed with pip

```
pip install cellml
```

### Install from source

Clone the repository to your computer

```
git clone https://github.com/caromccue/CellML.git
```

and install with pip 

```
cd CellML
pip install .
```

## Usage

### Quickstart

A time series of images of cells on surfaces must be stored in a directory prior to usage of CellML
The application can then be used to process the images as follows:
```
cellml process --save-plot path/to/directory
```

### `cellml process` command

The `process` command takes a directory of images, segments the cells in each image and determines how many cells are adherent or detached from the surface.
The program saves a `.csv` file at the root of that directory with the name of each image, the time it was taken (from EXIF data) and the number of cells (total, adherent and detached).

#### Arguments

- `-c`, `--check-segmentation` displays the result of segmenting an image (selected at approximately 80% of the time series) to verify that the segmentation algorithm works well before processing.
- `-o`, `--save-overlay` resaves all images in the directory with an overlay showing detected cells in red (adherent) or green (detached) for process control.
- `-p`, `--save-plot` generates and saves plots of cell detachment over time
- `-v`, `--verbose` increases the verbosity level

### `cellml segment` command

The `segment` command runs the segmentation algorithm on an image or a directory of images and saves the segmented cell images to disk.

#### Arguments

- `-o`, `--save-overlay` resaves all images in the directory with an overlay showing detected cells
- `-v`, `--verbose` increases the verbosity level
- `-p`, `--postsize` selects the size of microposts on the surfaces in the images to set the segmentation parameters accordingly

### `cellml train` command

The `train` command is used to train the machine learning models used to label the segmented cells as adherent or not. A directory of training data is expected containing subdirectories named `Detached` and `Adherent` containing grayscale images of segmented cells (use the `segment` command to generate the images).

#### Arguments

- `-m`, `--model` selects the type of model to train (svm|cnn|cnn-transfer)
- `-tb`, `--tensorboard` saves logs for tensorboard visualization in `<cwd>/logs`
- `-v`, `--verbose` increases the verbosity level

## Repository structure

- `models`: pre-trained machine learning models for cell adhesion discrimination
- `notebooks`: jupyer notebooks evaluating different image segmentation strategies
- `src`: source code for the project
    - `cell_processing`: processing pipeline from directory to detachment rate
    - `data`: data processing methods, including segmentation and extraction
    - `models`: model definition and training scripts for the cell binary labelling task
    - `visualization`: visualization and plotting methods
    - `cli.py`: entry point to the command line interface
- `tests`: unittesting

## License

This project is licensed under the GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details.

## Credit

Initial models were built starting from the example at:
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

Live data visualization class TrainingPlot originally from:
https://github.com/kapil-varshney/utilities/blob/master/training_plot/trainingplot.py
