""" just more indepth on the loss and optizer. they were making my training file look messy"""


def E(W, y_Dp, Zp, P):
    """
    Calculates the expectation E(W) this is the loss function used in the paper.
    it is just MSE loss. just laying it out here for reference.
    render this in markdown to see: \( E(W) = \frac{1}{P} \sum_{p=1}^{P} y_{Dp}(Z'^{(p)}, W) \)
    
    Parameters:
    W (Tensor): The parameters of the model.
    y_Dp (callable): The function that computes y for a given Z' and W.
    Zp (Tensor): The input data for which the expectation is being computed.
    P (int): The number of samples.
    
    Returns:
    Tensor: The computed expectation value.
    
    # Example usage (the following is just an example and will not run as is because y_Dp, Zp, and W need to be defined):
    # W = torch.randn(10)  # Example tensor for W
    # Zp = torch.randn(P, ... )  # Example tensor for Z'
    # P = 100  # Example value for P
    # expectation_value = E(W, y_Dp, Zp, P)
    # print(expectation_value)
    """
    # Initialize the summation result
    summation_result = 0.0
    
    # Loop over the range of P to compute the sum
    for p in range(1, P + 1):
        # Call y_Dp as a function with the current sample Zp[p-1] and W
        summation_result += y_Dp(Zp[p-1], W)
        
    # Compute the expectation by dividing the sum by P
    expectation = (1 / P) * summation_result
    return expectation



def sgd_(lr, net):
    """optimzier in paper
        goes like this: weight = weight - lr * gradient 
    """
    for f in net.parameters():
        f.data.sub_(f.grad.data * lr)
    return net