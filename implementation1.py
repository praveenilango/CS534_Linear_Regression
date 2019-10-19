import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read CSVs
def get_data():
    """
    Get training, validation and test data
    """
    train = pd.read_csv("./PA1_train.csv")
    dev = pd.read_csv("./PA1_dev.csv")
    test = pd.read_csv("./PA1_test.csv")
    return train, dev, test


# Add bias
def add_bias(df):
    """
    Add dummy variable to control intercept
    """
    df["dummy"] = 1
    return df


# Seperate Features from response
def seperate(df_train):
    """
    input: dataframe
    """
    # Grab all continuous features
    x = df_train.iloc[:, 0:-1]
    # Split dates
    x = split_date(x)
    x = x.drop(["date"], axis=1)
    # Grab response y
    y = df_train.iloc[:, -1]

    return x, y


# Add new features [Month, Day, Year]
def split_date(df_train):
    """
    splits date into seperate features
    input: dataframe
    """
    print("Splitting date...")
    for i in range(0, len(df_train)):
        df_train.loc[i, "month"] = int(df_train.loc[i, "date"].split("/")[0])
        df_train.loc[i, "day"] = int(df_train.loc[i, "date"].split("/")[1])
        df_train.loc[i, "year"] = int(df_train.loc[i, "date"].split("/")[2])
    print("Done")
    return df_train


# Normalize data
def normalize(df1):
    """
    Normalizes feature matrix
    input: feature df
    """
    print("Normalizing...")
    x = (df1 - np.min(df1)) / (np.max(df1) - np.min(df1))
    print("DONE")
    return x


# Linear regression function
def linear_regress(x, y, eta, t, lamb):
    """
    x: input/features
    y: opuput
    eta: learning rate
    t: iterations
    lamb: regularization constant
    """
    print("#Learning Rate : " + str(eta) + "#####")

    n = 0
    e = np.zeros(len(y))
    errors = []
    gradient = []

    # Initialize weights [w] and predictions [y_hat]
    w = np.zeros(len(x[0]))

    while n < t:
        # Initialize gradient for each epoch
        gradient_vector = np.zeros(len(x[0]))

        # Traverse through each data point
        for i in range(len(x)):
            # Predicted value
            y_hat = np.dot(w.T, x[i])

            # Error
            e[i] = ((y[i] - y_hat) ** 2)

            # Regularization
            if np.dot(w.T, w) == 0:
                r = 0
            else:
                r = (np.dot(w.T, w)) ** 0.5

            # Traverse through each feature to update corresponding weights
            for j in range(len(x[0])):
                gradient_vector[j] += ((-2) * (y[i] - y_hat) * x[i, j]) + (2 * lamb * r)

        # Update weights
        w -= eta * gradient_vector
        # Calculate SSE
        errors.append(sum(e))
        # Norm of gradient
        convergence_criteria = np.dot(gradient_vector.T, gradient_vector) ** 0.5
        gradient.append(convergence_criteria)

        ####
        # print(f'#####Iteration : {n+1}#####')
        # print(f'Gradient : {gradient[n]}')

        ####
        if (gradient[n] / (10 ** 9)) > 1 and (n + 1) <= 6:
            t = 8

        n += 1
        if convergence_criteria < 0.5:
        	print("#Iteration: " + str(n) +"####")
        	print("#Gradient: " + str(gradient[n-1]))

            #print(f'#Iteration : {n}#####')
            #print(f'Gradient : {gradient[n - 1]}')
            print("")
            print("")
            print("")
            return w, errors, gradient, n
        if (n) % 50 == 0:
            print("#Iteration: " + str(n) +"####")
        	print("#Gradient: " + str(gradient[n-1]))

    print("#Iteration: " + str(n) +"####")
    print("#Gradient: " + str(gradient[n-1]))

    print("")
    print("")
    print("")
    return w, errors, gradient, n

if __name__ == '__main__':
    #####DATA PREP#####

    # load csv
    df_train, df_dev, df_test = get_data()
    # Drop ID Feature
    df_train = df_train.drop("id", axis=1)

    # Grab features and Response
    x, y = seperate(df_train)

    # Normalize continuous features
    x_norm_df = normalize(x)
    # Add Bias
    x_norm_df = add_bias(x_norm_df)
    x_norm = x_norm_df.values

    learning_rates = [10**0, 10**-1]#, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7]
    lambdas = [0, 10**-3, 10**-2, 10**-1, 1, 10, 100]

    weights1, sse1, gradient1, iter1 = linear_regress(x_norm, y, 10 ** 0, 50000, 0)
    weights2, sse2, gradient2, iter2 = linear_regress(x_norm, y, 10 ** -1, 50000, 0)
    weights3, sse3, gradient3, iter3 = linear_regress(x_norm, y, 10 ** -2, 50000, 0)
    weights4, sse4, gradient4, iter4 = linear_regress(x_norm, y, 10 ** -3, 50000, 0)
    weights5, sse5, gradient5, iter5 = linear_regress(x_norm, y, 10 ** -4, 50000, 0)
    weights6, sse6, gradient6, iter6 = linear_regress(x_norm, y, 10 ** -5, 50000, 0)
    weights7, sse7, gradient7, iter7 = linear_regress(x_norm, y, 10 ** -6, 50000, 0)
    weights8, sse8, gradient8, iter8 = linear_regress(x_norm, y, 10 ** -7, 50000, 0)
    """
    for learning_rate in learning_rates:
        if learning_rate == 1:
            string_learning_rate = "1e0"
        elif learning_rate == 0.1:
            string_learning_rate = "1e-1"
        elif learning_rate == 0.01:
            string_learning_rate = "1e-2"
        elif learning_rate == 0.001:
            string_learning_rate = "1e-3"
        elif learning_rate == 0.0001:
            string_learning_rate = "1e-4"
        elif learning_rate == 0.00001:
            string_learning_rate = "1e-5"
        elif learning_rate == 0.000001:
            string_learning_rate = "1e-6"
        elif learning_rate == 0.0000001:
            string_learning_rate = "1e-7"

        weights, sse, gradient = linear_regress(x_norm, y, learning_rate, 2000, 0)
        plt.plot(sse)
        plt.title("SSE vs Iterations w/ Learning Rate "+ str(learning_rate) )
        plt.legend(["Normalized SSE - Training Data"])
        plt.xlabel('iterations')
        plt.ylabel('SSE')
        filename = "learning_rate_" + string_learning_rate + "_lambda_0"
        plt.savefig(filename)
    """