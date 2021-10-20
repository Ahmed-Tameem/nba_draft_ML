This project tests the accuracy of a naive logistic regression model for predicting which players from the NBA draft combine will end up playing in the NBA.

The model is trained on data from 2012 to 2015 and tested on the data from 2016. The model achives an accuracy of 75.4% for new data.

Before running main.py change the "path" variable in the function "scrape" to the full path of the data.

Dependencies: numpy, pandas, matplotlib, [nba_api](https://github.com/swar/nba_api).

# Credit to [Data World](https://data.world/achou/nba-draft-combine-measurements) for providing the data used.
# Credit to Professor [Brad Quinton](https://ece.ubc.ca/brad-quinton/) for providing the skeleton code for the logisitc regression model.
