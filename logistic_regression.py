import numpy as np
import matplotlib.pyplot as plt
from scrape import scrape

######################################################################################################
# Credit to https://data.world/achou/nba-draft-combine-measurements for providing the data           #
# Credit to professor Brad Quinton for providing the skeleton code for the logisitc regression model #
######################################################################################################


def sigmoid(x):
    """ A vectorized implementation of the Sigmoid Function
    
    Inputs:
        x -- A NumPy array, or a single float or int
        
    Returns:
        A NumPy array forthe Sigmoid function applied to x
    """

    s = 1/(1+np.exp(-x))
    
    return s

def initialize_parameters(n):
    """ Initialize the parameters with zeros for Logistic Regression
    
    Inputs:
        n: An int for number of input features
        
    Returns:
        NumPy Array: the W parameter vector of shape (n, 1)
        float: the b bias paramter
    """

    w = np.zeros((n,1))
    b = 0.0

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

    x1 =np.matmul(w.T,X)+b
    A = sigmoid(np.matmul(w.T,X)+b)
    
    return A

def compute_cost(A, Y):
    """ Vectorized Logistic Regression Cost Function
    
    Inputs:
        A: NumPy array of shape (1, m)
        Y: NumPy array (m, ) of known labels
        
    Returns:
        A single float for the cost
    """
    
    cost = float(np.sum(  np.matmul(np.log(A),Y)  +  np.matmul(np.log(1-A),1-Y)  ))/  (-1*(A.shape[1]))

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

    dw = (np.matmul(X,(A-Y).T))  /  (X.shape[1])
    db = np.sum(A-Y)  /  (X.shape[1])

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

    w, b = initialize_parameters(X.shape[0])

    costs = []
    
    for i in range(num_iterations):
        A = hypothesis(X,w,b)
        dw, db = compute_gradients(A, X, Y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        cost = compute_cost(A, Y)
        costs.append(cost)
        
        # Print cost after ever 5000 iterations
        if print_costs and i % 5000 == 0:
            print("Iteration {0} - Cost: {1}".format(i, str(costs[-1])))

    return w, b, costs

def train(X, Y):
    """ Use the Logistic Regression Model to make predictions
    
    Inputs:
        X: NumPy array (n, m) of feature data
        Y: NumPy array (m, ) of known labels
    
    Returns:
        w: NumPy array of parameters with shape (n, 1)
        b: float for the bias parameter
        costs: Float list of all the costs associated with each iteration
    """

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
    y_pred = np.rint(hypothesis(X, w, b))

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
    accuracy = np.sum(Y == predict(X, w, b))/X.shape[1]

    return accuracy

def plot_costs(costs):
    """ Plot the cost as a function of the number of iterations
    Inputs:
        costs: Float list of all the costs associated with each iteration
        
    Returns:
        X: NumPy array of input samples of shape (n, m)
        Y: NumPy array of the labels of shape (m, )
    """

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
