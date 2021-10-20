from nba_api.stats.static import players
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

######################################################################################################
# Credit to https://data.world/achou/nba-draft-combine-measurements for providing the data           #
# Credit to professor Brad Quinton for providing the skeleton code for the logisitc regression model #
######################################################################################################

def scrape():
    players_in_nba = players.get_players()

    players_in_nba_names = [player["full_name"] for player in players_in_nba]


    path = "D:\\ahmad\\Studies\\Courses\\CPEN 400D\\Misc\\NBA_draft\\nba_draft_ML\\data\\nba_draft_combine_all_years.csv"
    with open(path, 'r') as data:
        df = pd.read_csv(data)



    df_cleaned = df.drop(df.columns[[0,1,2,3]], axis = 1, inplace = False)

    for column in df_cleaned:
        df_cleaned[column].fillna(df_cleaned[column].mean(), inplace = True)

    X = df_cleaned.to_numpy().T

    Y = list()
    for player in df["Player"]:
        Y.append(1 if(player in players_in_nba_names) else 0)
    Y = np.array(Y)

    return X, Y

def sigmoid(x):
    """ A vectorized implementation of the Sigmoid Function
    
    Inputs:
        x -- A NumPy array, or a single float or int
        
    Returns:
        A NumPy array forthe Sigmoid function applied to x
    """

    ### START CODE HERE ### (~ 1 line of code) ###
    # YOUR CODE HERE
    s = 1/(1+np.exp(-x))
    ### END CODE HERE ###
    
    return s

def initialize_parameters(n):
    """ Initialize the parameters with zeros for Logistic Regression
    
    Inputs:
        n: An int for number of input features
        
    Returns:
        NumPy Array: the W parameter vector of shape (n, 1)
        float: the b bias paramter
    """
    ### START CODE HERE ### (~ 2 line of code)
    # YOUR CODE HERE
    w = np.zeros((n,1))
    b = 0.0
    ### END CODE HERE ###

    assert(w.shape == (n, 1))
    assert(isinstance(b, float))
    
    return w, b

def hypothesis(X, w, b):
    """ 
    Inputs:
        X: NumPy array of input samples of shape (n, m)
        w: NumPy array of parameters with shape (n, 1)
        b: float for the bias parameter
        
    Returns:
        NumPy array of shape (1, m) with the hypothesis of each sample
    """
    A = None
    
    ### START CODE HERE ### (~ 1 line of code)
    # YOUR CODE HERE
    x1 =np.matmul(w.T,X)+b
    A = sigmoid(np.matmul(w.T,X)+b)
    ### END CODE HERE ###
    
    return A

def compute_cost(A, Y):
    """ Vectorized Logistic Regression Cost Function
    
    Inputs:
        A: NumPy array of shape (1, m)
        Y: NumPy array (m, ) of known labels
        
    Returns:
        A single float for the cost
    """
    
    cost = None
    
    ### START CODE HERE ### (~ 1-5 line of code)
    # YOUR CODE HERE
    cost = float(np.sum(  np.matmul(np.log(A),Y)  +  np.matmul(np.log(1-A),1-Y)  ))/  (-1*(A.shape[1]))

    ### END CODE HERE ###

    return cost

def compute_gradients(A, X, Y):
    """ Compute the gradients of the cost function 
    
    Inputs:
        A: NumPy array of shape (1, m)
        X: NumPy array of shape (n, m)
        Y: NumPy array of shape (m, )
    
    Returns:
        Two NumPy arrays. One for the cost derivative w.r.t. dw
        and one for the cost derivative w.r.t. db
    """
    dw = None
    db = None
    
    ### START CODE HERE ### (~ 2-3 line of code)
    # YOUR CODE HERE
    dw = (np.matmul(X,(A-Y).T))  /  (X.shape[1])
    db = np.sum(A-Y)  /  (X.shape[1])
    ### END CODE HERE ###

    assert(dw.shape == (X.shape[0], 1))

    return dw, db

def gradient_descent(X, Y, num_iterations, learning_rate, print_costs=True):
    """ Perform Gradient Descent for Logistic Regression
    
    Inputs:
        X: NumPy array (n, m)
        Y: NumPy array (m,)
        num_iterations: int for number of gradient descent iterations
        learning_rate: float for gradient descent learning rate
        print_costs: bool to enable printing of costs
    
    Returns:
        w: NumPy array for trained parameters w
        b: float for trained bias parameter b
        costs: Python list of cost at each iteration
    """
    w, b = None, None

    ### START CODE HERE ### (~1  line of code)
    # YOUR CODE HERE
    w, b = initialize_parameters(X.shape[0])
    ### END CODE HERE ###
    
    # We will use a list to store the cost at each iteration
    # so that we can plot this later for educational purposes
    costs = []
    
    for i in range(num_iterations):

        ### START CODE HERE ### (~5  line of code)
        # YOUR CODE HERE
        A = hypothesis(X,w,b)
        dw, db = compute_gradients(A, X, Y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        cost = compute_cost(A, Y)
        ### END CODE HERE ###
        
        # Convert and save the cost at this iteration
        costs.append(cost)
        
        # Print cost after ever 5000 iterations
        if print_costs and i % 5000 == 0:
            print("Iteration {0} - Cost: {1}".format(i, str(costs[-1])))

    return w, b, costs

def train(X, Y):
    w, b, costs = gradient_descent(X, Y, 15000, 0.0000000113)
    return w, b, costs

def predict(X, w, b):
    """ Use the Logistic Regression Model to make predictions
    
    Inputs:
        X: NumPy array (n, m) of feature data
        w: NumPy array (n, 1) of trained parameters w
        b: float for trained bias parameter
    
    Returns:
        NumPy array (m, ) of predictions.  Values are 0 or 1.
    """
    y_pred = None

    ### START CODE HERE ### (~1  line of code)
    # YOUR CODE HERE
    y_pred = np.rint(hypothesis(X, w, b))
    ### END CODE HERE ###
    
    return y_pred

def compute_accuracy(X, Y, w, b):
    """ Compute the accuracy of a trained Logistic Regression 
        model described by its trained parameters

    Inputs:
        X: NumPy array (n, m) feature data
        Y: NumPy array (m, ) known labels for feature data X
        w: NumPy array (n, 1) trained model parameters w
        b: float for trained bias parameter
    
    Returns:
        float between 0 and 1 denoting the accuracy of the
        Logistic Regression model 
    """
    accuracy = None
    
    ### START CODE HERE ### (1-2  line of code)
    # YOUR CODE HERE
    accuracy = np.sum(Y == predict(X, w, b))/X.shape[1]
    ### END CODE HERE ###
    return accuracy

def plot_costs(costs):
    plot2 = plt.figure(2)
    plt.plot(costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title("Logistic Regression Training Cost Progression")
    plt.show(block = False)

X, Y = scrape()

X_train = X.T[:456].T
X_test = X.T[456:].T

Y_train = Y[:456]
Y_test = Y[456:]

w, b, costs = train(X_train, Y_train)
accuracy = compute_accuracy(X_test, Y_test, w, b)
print('Accuracy: {0}%'.format(accuracy*100))

plot_costs(costs)
plt.show()
