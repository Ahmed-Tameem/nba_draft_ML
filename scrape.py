from nba_api.stats.static import players
import pandas as pd
import numpy as np

def scrape():
    """ Fetch and clean the data to be used in training
    Inputs:
        None
        
    Returns:
        X: NumPy array of input samples of shape (n, m)
        Y: NumPy array (m, ) of known labels
    """

    players_in_nba = players.get_players()

    players_in_nba_names = [player["full_name"] for player in players_in_nba]   #fetch the names of all the players playing in the NBA


    path = "D:\\ahmad\\Studies\\Courses\\CPEN 400D\\Misc\\NBA_draft\\nba_draft_ML\\data\\nba_draft_combine_all_years.csv"   #Change to path to data
    with open(path, 'r') as data:
        df = pd.read_csv(data)

    df_cleaned = df.drop(df.columns[[0,1,2,3]], axis = 1, inplace = False)  #remove columns with features not used to train such as the player name
    for column in df_cleaned:
        df_cleaned[column].fillna(df_cleaned[column].mean(), inplace = True)    #Fill missing data with the mean of that feature for all players

    X = df_cleaned.to_numpy().T

    Y = list()
    for player in df["Player"]:
        Y.append(1 if(player in players_in_nba_names) else 0)   #Check if the player plays in the NBA to create  labels for the data
    Y = np.array(Y)

    return X, Y