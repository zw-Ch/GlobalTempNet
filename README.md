# ResGraphNet
ResGraphNet is a deep neural network used to predict time series. It effectively combines Graph Neural Network(GNN) with ResNet Module, so it has a stronger performance than traditional time series prediction methods.<br>
The paper is available in <br>

## Installation
ResGraphNet is based on [Pytorch](https://pytorch.org/docs/stable/index.html) and [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)<br>
Firstly please create a virtual environment for yourself<br>
`conda create -n your-env-name python=3.9`<br><br>
Then, there are some Python packages need to be installed<br>
`conda install pytorch torchvision torchaudio cudatoolkit=11.3`<br>
`conda install pyg -c pyg`<br>
`conda install matplotlib`<br>

## Files / Folders Description
The main program of this project is [rnn_gnn_machine.py](https://github.com/czw1296924847/ResGraphNet/blob/main/rnn_gnn_machine.py), so you can run the program by typing the following command<br>
`python rnn_gnn_machine.py`<br><br>
The algorithm program [cal.py]() contains some custom functions required for the operation of this project.<br><br>
The preprocessor program [data_preprocess.py]() is used to sample the datasets in order to speed up the operation.<br><br>
The folder [dataset](https://github.com/czw1296924847/ResGraphNet/tree/main/datasets) is used to store the time series data to be predicted, and the default file format of data is .npy. Of course, you can change the data address in the 41th or 42th line in [rnn_gnn_machine.py](https://github.com/czw1296924847/ResGraphNet/blob/main/rnn_gnn_machine.py).<br><br>
The folder [result]() saves the prediction results of different time series data files based on different models

## Parameter Selection
Program [rnn_gnn_machine.py](https://github.com/czw1296924847/ResGraphNet/blob/main/rnn_gnn_machine.py) contains many parameters that can be changed, such as<br>
`gnn_style`: The type of GNN model style<br>
`gnn_style_all`: All the GNN model you can choose, where 'sage_res' as the model ResGraphNet that we have proposed in our paper, and others are the models based on traditional GNN layers<br>
`epochs`: The number of iteration rounds<br>

