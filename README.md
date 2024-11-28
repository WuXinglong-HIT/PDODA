# PDODA

## :house: Overview
In this work, we propose PDODA, a Propagation Depth Oriented Data Augmentation Architecture based on Grpah Neural Networks for Recommendation

## :hammer: Package
* torch==1.11
* numpy=1.22
* pandas
* sklearn
* tensorboardX

## :pencil: Results
The overall performance is shown below:
![Overall Performance](fig/performance.png)

## :triangular_flag_on_post: Additional Instructions
If an error, such as the one shown below, occurs during program execution, it is recommended to update the numpy version to 1.22.0 through the command "pip install numpy==1.22.0".

The error message reads as follows:
> "ValueError: setting an array element with a sequence. The resulting array exhibits inhomogeneity after a single dimension, with the specific detected shape being (90,) along with an inhomogeneous portion."

If error as shown below is reported during running, just change the numpy version to 1.22.0 (pip install numpy==1.22.0)
> ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (XX,) + inhomogeneous part.
