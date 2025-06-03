# Step-by-step Guide to Run the Pipeline

### Pipeline functionality

This pipeline is created to reproduce results for section 5.1 (`Structural Inductive Bias in Neural Circuits`). Specifically, this pipeline includes:
1. Train RNNs with varied initial weight rank (via SVD) for cognitive tasks in `neurogym` package (tasks supported are `CXT`, `2AF`, `DMS`, see [documentation](https://neurogym.github.io/neurogym/latest/) for more details).
2. Run various analysis to measure the degree of feature learning during model training, including weight change, activation similarity, representation alignment, kernel alignment, and manifold capacity and associated manifold geometry measurements.

### Pipeline structure
The pipeline has the following components:
1. `generate_input_params.ipynb`: generate the input parameters `.json` file for the pipeline
2. `feature_learning_rnn.sh`: `.sh` file to execute the `python` script, given the path to the parameter file.
3. `feature_learning_rnn.py`: The main `python` script to set up the `neurogym` environment, train the RNNs, and run the above analysis.
4. `visualization_notebook.ipynb`: Notebook to process and plot the results.

### How to run the pipeline step-by-step
1. Run the notebook `generate_input_params.ipynb` to generate the `json` parameter file.
2. Run the bash script `feature_learning_rnn.sh` to run the main python script (remember to change `-p` to the path to the `json` file that you just created).
3. Run the `visualization_notebook.ipynb` to process and plot the results.