This project tests the accuracy of a naive logistic regression model and a DNN for predicting which players from the NBA draft combine will end up playing in the NBA.

The models are trained on data from 2012 to 2015 and tested on the data from 2016.

The logistic regression model achieves a validation accuracy of 75.4% and the DNN achieves a validation accuracy of 77.9% (80.3% training accuracy).

Before running "DNN.py" or "logistic_regression.py", Change the "path" variable in the function "scrape" to the full path of the data

Dependencies: numpy, pandas, matplotlib, tensorflow, [nba_api](https://github.com/swar/nba_api).

# Credit to [Data World](https://data.world/achou/nba-draft-combine-measurements) for providing the data used.
# Credit to Professor [Brad Quinton](https://ece.ubc.ca/brad-quinton/) for providing the skeleton code for the logistic regression model.
