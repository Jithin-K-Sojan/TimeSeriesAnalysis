# TimeSeriesAnalysis
This is a predictive model on the NC-94 crop dataset. The files climate.txt and crop.txt were used to predict the crop yields of clusters of counties.

The clustering was done based on four parameters in the climate.txt file: MaxTemp, MinTemp, Radiant, and Precipitation.

After obtaining clusters, the crop yields of each of the clusters were predicted using moving average regression with a window size of k. Exponential smoothing was used as well.

For furhter details and to see the results of the run, look at the docx file in the folder.
