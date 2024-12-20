# K-means clustering for GPU

This project is part of GPGPU course on Faculty of Mathematics and Information Science at Warsaw University of Technology.

## Run this project
To build the project run:
```{Bash}
mkdir build
cd build
cmake ..
make
```
And then to run:
```{Bash}
./KMeans
```
In file main.cpp you can alter parameters of the run, like number of points in each class, or number of classes.
The first plot you will see is just data generated. The second one is starting position of centroids.
The next one is the result of the algorithm.
Plots will be displayed twice, for GPU and for CPU.

## Results
In main.cpp you can find function *create_data_for_experiments*, which runs a lot of experiments. The results are saved in *./build/result.txt*, so you need to move it to ./ploting and change to *./csv* file. In *./ploting/plots/* you can find the results.

## Varying number of features
K-Means algorithm works for number of features not known at compiletime, however data creation requires knowing the number of features at compiletime. Also the code needs to be restructured, as displaying works only for two features.