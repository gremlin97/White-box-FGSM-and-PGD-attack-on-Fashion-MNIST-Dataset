# Read Me

## Folder Contents
* The codebase consists of both a Colab Jupyter Notebook and a python file for execution. Both give similar results, and the only reason to use Colab again was for faster prototyping and debugging using a GPU. Model has the model version saved using `torch.save(model.pt)`, Graphs has the graphs saved during execution, and Latex has the final latex submission. The console outputs are in the `Output.txt` file.

## How to Run?

### Python File
* Run the `train_fmnist.py` file to train, test and attack the model using PGD and FGSM.
* The Model plots (val and train loss, PGD Images) will be saved in the same directory.
* Check the console for the the accuracy, loss and attack outputs.

### IPYNB File

* Run this link in Colab or run the IPYNB file from the jupyter notebook in your local instance.