# keras-plotter

Keras Plotter is a Python library designed to visualize neural networks created with Keras. It is particularly useful for models with two inputs receiving competitive stimuli but is versatile enough to support various neural network architectures.

It was originally made to [visualize neural networks receiving competitive stimuli](https://www.researchgate.net/publication/383912312_Visualizing_Information_in_Deep_Neural_Networks_Receiving_Competitive_Stimuli) using the MNIST dataset. For this reason, the tutorial follows this context.

![Plot showing neural network's activations to input.](results/images/reference_plot.png)

## Dependencies

* python 3.10+
* tensorflow v2.15.0.post1
* keras v2.15.0
* matplotlib v3.8.3+
* numpy v1.26.4+
* attrs v23.2.0+

## Usage

An example of usage can be found in `main.ipynb`.

First, it is necessary to clone the repository and install its dependencies.

```bash

git https://github.com/hugobbi/keras-plotter.git
cd keras-plotter
pip install -e .

```

## Generating dataset

The dataset can be generated using the `Dataset` class located in the dataset module. The method `build_vf_dataset` builds the visual fields dataset based upon the train and test sets passed to the `Dataset` class.

### Visualizing dataset

The `Utils` module has various functions to visualize the dataset. The function `show_dataset` can be used to display `n` entries. To display instances containing a specific digit, the `display_n_digits` function can be used.

## Selecting a model

A Keras neural network model can be created and trained from scratch or loaded from a file.

### Model definitions

Some model definitions can be found in the `main.ipynb` file.

### Double visual fields model

In order to create and plot a double visual fields model, the left and right visual fields' layers must be added to the model alternately.

### Training

For the purpose of saving RAM, it is recommended to use the `data_generator` functions (specifying single or double visual field network functions) located in the `Utils` module.

### Saving model to file

It is possible to save the trained model to the `models/` path, using the `tf.keras.Model.save` method.

### Loading model from file

The models are stored in the `models/` directory. It is possible to load them using the `tf.keras.models.load_model` method.

## Plotting Keras neural networks

The plotting of neural newtorks is done using the `NeuralNetworkPlotter` class, located in the `Plotting Neural Networks` module. It receives a trained model to be used for plotting.

Plotting is done using the `plot` method in the `NeuralNetworkPlotter` class. It receives an input and displays the trained neural network's activations and weights for that specific input, also displaying the neural network's architecture.

When the function is executed, an image of the plot will be stored in `results/images/`.

The following layer types are currenty supported by the plotter:
* Input [x]
* Dense layers [x]
* Concatenate layers [x]
* Convolutional layers [x]
* Max Pooling [] 

### Choosing input for visualization

The function `display_n_digits`, located in the `Utils` module can be used to choose an input for the plotter. It displays `n` instances of the digit in the dataset, as well as its index, which is then be passed to the plotter.

### Attribute lenses

Attribute lenses are used to visualize which classes are being represented inside each trained hidden layer of the model for an input. If generated, the top `k` classes are displayed in the neural network plot. In order to generate and train these attribute lenses, the methods `generate_attribute_lenses` and `train_attribute_lenses` from the `NeuralNetworkPlotter` class are used. It is recommended to use the same training parameters in `train_attribute_lenses` as was used to train the neural network for better results.

### Saving plotters

The `NeuralNetworkPlotter` object can be saved to a file and also loaded from one with the `save_obj` and `load_obj` functions located in the `Utils` module.

## Metrics

The `Metrics` module contains various functions to evaluate the network's internal representations, as used [here](https://www.researchgate.net/publication/383912312_Visualizing_Information_in_Deep_Neural_Networks_Receiving_Competitive_Stimuli).

### Computing cosine similarity matrices

In order to compute the cosine similarity matrix (CSM) for every layer of the model, it is first necesary to compute the prototype for each digit for each layer. This is done by using the `generate_prototypes` function. It is recommended to use the `generate_prototypes_mp` function, as it uses all cores of the CPU to compute the prototypes in parallel.

To finnaly compute the CSMs, use the `compute_cosine_similarity_matrix` function.

### Visualizing CSM

To visualize the matrices, use the `plot_csm` function. The alternative `plot_csm_interactively` plots the matrix in the same way, the difference being the user can
choose from an interactive dropbar which layer to plot the CSM from. It is important to note that the CSM is symmetrical, and the values on the lower left half are not
computed. To represent them, they are colored black by default, but this can be changed using the `color_not_computed` parameter.

## Computing orthogonality

Calculating the orthogonality measure can be done using the `compute_orthogonality` function.

## Useful Utils functions

* `show_dataset`: displays `n` instances of a dataset.
* `show_instance`: shows a single instance of a dataset.
* `display_n_digits`: displays `n` instances of a digit in the dataset.
* `split_array`: splits an array given a percentage (used for train/test split)
* `save_obj`: saves a Python object to a file.
* `save_obj`: loads a Python object from a file.
* `data_generator_dvf`: generator used to train or test a double visual fields model in batches.
* `data_generator_svf`: generator used to train or test a single visual field model in batches.
