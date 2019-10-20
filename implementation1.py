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


# Separate Features from response
def separate(df_train):
    """
    input: dataframe
    """
    # Grab all continuous features
    x = df_train.iloc[:, 0:-1]
    # Split dates
    x = split_date(x)
    x = x.drop(["date"], axis=1)
    # Grab response y
    y = df_train.iloc[:, -1].values

    return x, y


# Add new features [Month, Day, Year]
def split_date(df_train):
    """
    splits date into separate features
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
def linear_regress(x, y, x_val, y_val, eta, t, lamb, normalized):
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
    errors_val = []
    gradient = []

    # Initialize weights [w] and predictions [y_hat]
    w = np.zeros(len(x[0]))

    cur_grad = 0
    prev_grad = 0

    while n < t:
        # Initialize gradient for each epoch
        gradient_vector = np.zeros(len(x[0]))

        # y_hat = np.matmul(w.T, x)
        y_hat = np.matmul(x, w)
        e = (y - y_hat) ** 2

        reg = np.array([i for i in w])
        reg[0] = 0
        gradient_vector = (-2) * np.matmul(x.T, (y - y_hat)) + 2*lamb*reg

        # Calculate SSE
        errors.append(sum(e))
        errors_val.append(get_sse(x_val, y_val, w))

        # Update weights
        w -= eta * gradient_vector
        # Norm of gradient
        convergence_criteria = np.dot(gradient_vector.T, gradient_vector) ** 0.5
        gradient.append(convergence_criteria)

        ####
        # print(f'#####Iteration : {n+1}#####')
        # print(f'Gradient : {gradient[n]}')

        ####
        if (gradient[n] / (10 ** 9)) > 1 and (n + 1) <= 6 and normalized == True:
            t = 8
        #elif (gradient[n] / (10 ** 80)) > 1 and (n + 1) <= 6 and normalized == False:
        #    t = 8

        n += 1
        if convergence_criteria < 0.5:
            print("#Iteration: " + str(n) +"####")
            print("#Gradient: " + str(gradient[n-1]))

            #print(f'#Iteration : {n}#####')
            #print(f'Gradient : {gradient[n - 1]}')
            print("")
            print("")
            print("")
            return w, errors, errors_val, gradient, n
        if (n) % 50000 == 0:

            print("#Iteration: " + str(n) +"####")
            print("#Gradient: " + str(gradient[n-1]))

    print("#Iteration: " + str(n) +"####")
    print("#Gradient: " + str(gradient[n-1]))

    return w, errors, errors_val, gradient, n

#Validate with validation data
def get_sse(x, y, w):
    """
    :param x: input
    :param y: output
    :param w: weight from training model
    :return: errors - sse
    """

    y_hat = np.matmul(x, w)
    e = (y - y_hat)**2
    return sum(e)

#Statistics on Numerical
def get_stats(df1, categoical_list):
    df1.describe().transpose().drop(columns=['count','25%', '50%', '75%']).drop(categoical_list).to_csv('training_stats.csv')

#Frequency Count on Categorical
def get_freq_percentage_count(df1, col_names):
    for col in col_names:
        percentage = (df1[col].value_counts(normalize=True).rename_axis('values').to_frame('freq_percentage').sort_index() * 100).round(1).astype(str) + '%'
        freq = df1[col].value_counts().rename_axis('values').to_frame('freq_count')
        result = pd.concat([percentage, freq], axis=1, join='inner')
        result.to_csv(col+"_frequency.csv")

if __name__ == '__main__':
    #####DATA PREP#####

    # load csv
    df_train, df_dev, df_test = get_data()
    # Drop ID Feature
    df_train = df_train.drop("id", axis=1)
    df_validation = df_dev.drop("id", axis=1)

    # Grab features and Response
    x, y = separate(df_train)
    x_val, y_val = separate(df_validation)

    x_values = x.values
    y_values = y
    x_val_values = x_val.values
    y_val_values = y_val

    # Normalize continuous features
    x_norm_df = normalize(x)
    x_val_norm_df = normalize(x_val)
    features_name = x_norm_df.columns

    # Add Bias
    x_norm_df = add_bias(x_norm_df)
    x_norm = x_norm_df.values
    x_val_norm_df = add_bias(x_val_norm_df)
    x_val_norm = x_val_norm_df.values

    learning_rates = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7]
    lambdas = [0, 10**-3, 10**-2, 10**-1, 1, 10, 100]

    get_stats(df_train, ['waterfront', 'grade', 'condition'])
    get_freq_percentage_count(df_train, ['waterfront', 'grade', 'condition'])


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

        weights, sse, sse_val_list, gradient, iterations_count = linear_regress(x_norm, y, x_val_norm, y_val, learning_rate, 500000, 0, True)
        sse_val = get_sse(x_val_norm, y_val,weights)
        filename = "Normalized_learning_rate_" + string_learning_rate + "_lambda_0"
        print("learning rate: " + str(learning_rate))
        print("lambda: " + str(0))
        print("SSE training: " + str(sse[-1]))
        print("SSE validation: " + str(sse_val))
        print("Iteration Count: " + str(iterations_count))
        pd.DataFrame(weights, features_name).to_csv("./weights/"+filename+".csv")
        print("weight produced at: " + "./weights/"+"weights_"+filename+".csv")
        print()

        plt.plot(sse, label = "Normalized SSE - Training Data", color='blue')
        plt.plot(sse_val_list, label = "Normalized SSE - Validation Data", color='red')
        plt.legend(["Normalized SSE - Training Data", "Normalized SSE - Validation Data"], loc = 1)
        plt.title("SSE vs Iterations w/ Learning Rate: " + str(learning_rate) + " & Lambda: " + str(0))
        plt.xlabel('iterations: ' + str(iterations_count))
        plt.ylabel('SSE')
        plt.savefig("./plots/"+"Normalized_"+filename)

def plot(sse, sse_val_list, legends, learning_rate, lamb, iterations_count, filename):
    plt.plot(sse, color='blue')
    plt.plot(sse_val_list, color='red')
    plt.legend(legends, loc=1)
    plt.title("SSE vs Iterations w/ Learning Rate: " + str(learning_rate) + " & Lambda: " + str(lamb))
    plt.xlabel('iterations: ' + str(iterations_count))
    plt.ylabel('SSE')
    plt.savefig("./plots/" + filename)


    for lamb in lambdas:
        if lamb == 1:
            string_lambdas = "1e0"
        elif lamb == 0:
            string_lambdas = "0"
        elif lamb == 0.001:
            string_lambdas = "1e-3"
        elif lamb == 0.01:
            string_lambdas = "1e-2"
        elif lamb == 0.1:
            string_lambdas = "1e-1"
        elif lamb == 10:
            string_lambdas = "1e1"
        elif lamb == 100:
            string_lambdas = "1e2"

        weights, sse, sse_val_list, gradient, iterations_count = linear_regress(x_norm, y, x_val_norm, y_val, 10**-5, 500000, lamb, True)
        sse_val = get_sse(x_val_norm, y_val, weights)
        filename = "learning_rate_1e5" + "_lambda_" + string_lambdas

        print("learning rate: " + str(10**-5))
        print("lambda: " + str(lamb))
        print("SSE training: " + str(sse[-1]))
        print("SSE validation: " + str(sse_val))
        print("Iteration Count: " + str(iterations_count))
        pd.DataFrame(weights, features_name).to_csv("./weights/"+filename+".csv")
        print("weight produced at: " + "./weights/"+"weights_"+filename+".csv")
        print()

        plt.plot(sse, label = "Normalized SSE - Training Data", color='blue')
        plt.plot(sse_val_list, label = "Normalized SSE - Validation Data", color='red')
        plt.legend(["Normalized SSE - Training Data", "Normalized SSE - Validation Data"], loc = 1)
        plt.title("SSE vs Iterations w/ Learning Rate: " + str(10**-5) + " & Lambda: " + str(lamb))
        plt.xlabel('iterations: ' + str(iterations_count))
        plt.ylabel('SSE')
        
        plt.savefig("./plots/"+"Normalized_"+filename)
"""
    #### NON NORMALIZED X ###
    learning_rates_non_normalized = [1, 0, 10**-3, 10**-6, 10**-9, 10**-15]
    for learning_rate in learning_rates_non_normalized:
        if learning_rate == 1:
            string_learning_rate = "1e0"
        elif learning_rate == 0:
            string_learning_rate = "0"
        elif learning_rate == 0.001:
            string_learning_rate = "1e-3"
        elif learning_rate == 0.000001:
            string_learning_rate = "1e-6"
        elif learning_rate == 0.000000001:
            string_learning_rate = "1e-9"
        elif learning_rate == 0.000000000000001:
            string_learning_rate = "1e-15"

        weights, sse, sse_val_list, gradient, iterations_count = linear_regress(x_values, y_values, x_val_values, y_val_values,
                                                                                learning_rate, 10000, 0, False)
        sse_val = get_sse(x_val_values, y_val_values, weights)
        filename = "learning_rate_" + string_learning_rate + "_lambda_0"
        print("learning rate: " + str(learning_rate))
        print("lambda: " + str(0))
        print("SSE training: " + str(sse[-1]))
        print("SSE validation: " + str(sse_val))
        print("Iteration Count: " + str(iterations_count))
        pd.DataFrame(weights, features_name).to_csv("./weights/" + filename + ".csv")
        print("weight produced at: " + "./weights/" + "weights_" + filename + ".csv")
        print()

        plt.plot(sse, label="Non-Normalized SSE - Training Data", color='blue')
        plt.plot(sse_val_list, label="Non-Normalized SSE - Validation Data", color='red')
        plt.legend(["Non-Normalized SSE - Training Data", "Non-Normalized SSE - Validation Data"], loc=1)
        plt.title("SSE vs Iterations w/ Learning Rate: " + str(learning_rate) + " & Lambda: " + str(0))
        plt.xlabel('iterations: ' + str(iterations_count))
        plt.ylabel('SSE')
        plt.savefig("./plots/" + "Non_Normalized_" + filename)
"""
