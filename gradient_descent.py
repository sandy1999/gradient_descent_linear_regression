import numpy as np # for matrix maths 

def compute_error_for_points(b, m, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0 # initial b gradient 
    m_gradient = 0 # initial m gradient
    N = float(len(points)) # number of points

    for i in range(0, len(points)):
        x = points[i, 0] # X values 
        y = points[i, 1] # y values 

        # computing gradients 
        b_gradient += -(2/N) * (y- (m_current * x)+ b_current)
        m_gradient += -(2/N) * x * (y- (m_current * x)+ b_current)
    new_b = b_current - (learning_rate * b_gradient) # converging to b
    new_m = m_current - (learning_rate * m_gradient) # converging to m
    return new_b, new_m

def gradient_descent_runner(b_starting, m_starting, points, learning_rate, num_iterations):
    b = b_starting # guess b  
    m = m_starting # guess m 

    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]
def main():
    # generate data points
    points = np.genfromtxt('data.csv',delimiter=',')

    # defining hyper parameters 
    learning_rate = 0.0001
    num_iterations = 1000

    # initial values (slope parameters)
    initial_b = 0 # intercept value guess
    initial_m = 0 # slope value guess
    print("Gradient Descent with initial b = {}, initial m = {} and initial error = {}".format(initial_b,initial_m, compute_error_for_points(initial_b, initial_m, points)))
    print("Running")
    [b, m] = gradient_descent_runner(initial_b, initial_m, points, learning_rate, num_iterations) 
    print("Gradient Descent after {} iterations is b = {}, m = {}, error = {}".format(num_iterations, b, m, compute_error_for_points(b, m, points)))
if __name__ == '__main__':
    main()