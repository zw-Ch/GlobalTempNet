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
The main program of this project is [running.py](https://github.com/czw1296924847/ResGraphNet/blob/main/running.py), so you can run the program by typing the following command<br>
`python rnn_gnn_machine.py`<br><br>
The algorithm program [cal.py](https://github.com/czw1296924847/ResGraphNet/blob/main/func/cal.py) contains some custom functions required for the operation of this project.<br><br>
The folder [dataset](https://github.com/czw1296924847/ResGraphNet/tree/main/datasets) is used to store the time series data to be predicted, and the default file format of data is .npy. Of course, you can change the data address in the 45th line in [running.py](https://github.com/czw1296924847/ResGraphNet/blob/main/running.py).<br><br>
The folder [result](https://github.com/czw1296924847/ResGraphNet/tree/main/result) saves the prediction results of different time series data files based on different models

## Parameter Selection
Program [running.py](https://github.com/czw1296924847/ResGraphNet/blob/main/running.py) contains many parameters that can be changed, such as<br>
`gnn_style`: The type of GNN model style<br>
`gnn_style_all`: All the GNN model you can choose, where 'sage_res' as the model ResGraphNet that we have proposed in our paper, and others are the models based on traditional GNN layers<br>
