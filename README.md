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
The running programs for different models is in the foloder [runn](https://github.com/czw1296924847/ResGraphNet/tree/main/run), so you can run the program by typing the following command<br>
`python running.py`<br><br>
You can use the spiral chart in [plot_result.py](https://github.com/czw1296924847/ResGraphNet/blob/main/plot_result.py) to draw the result of [running.py](https://github.com/czw1296924847/ResGraphNet/blob/main/running.py), and you can change the i1, i2 to choose the result data.<br><br>
The algorithm program [cal.py](https://github.com/czw1296924847/ResGraphNet/blob/main/func/cal.py) contains some custom functions required for the operation of this project.<br><br>
The folder [dataset](https://github.com/czw1296924847/ResGraphNet/tree/main/datasets) is used to store the time series data to be predicted, and the default file format of data is .npy. Of course, you can change the data address in the 50th line in [running.py](https://github.com/czw1296924847/ResGraphNet/blob/main/running.py).<br><br>
The folder [result](https://github.com/czw1296924847/ResGraphNet/tree/main/result) saves the prediction results of different time series data files based on different models
