

import numpy as np


import matplotlib.pyplot as plt

from myRLS import myRLS


class Regression:

    #def __init__(self):
        # plt.style.use('seaborn')
        # mpl.rcParams['font.family'] = 'serif'
    def f(self, x):
        return np.sinc(x) #np.sin(x)*np.cos(x) + 0.5 * x**3

    def create_plot(self, x, y, styles, labels, axlabels):
        plt.figure(figsize=(10, 6))
        for i in range(len(x)):
            plt.plot(x[i], y[i], styles[i], label=labels[i])
            plt.xlabel(axlabels[0])
            plt.ylabel(axlabels[1])
            plt.legend(loc=0)

    def create(self):
        x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
        self.create_plot([x], [self.f(x)], ['b'], ['f(x)'], ['x', 'f(x)'])
        res = np.polyfit(x, self.f(x), deg=1, full=True)
        ry = np.polyval(res[0], x)
        ry = np.polyval(res[0], x)
        #return ry
        self.create_plot([x, x], [self.f(x), ry], ['b', 'r.'], ['f(x)', 'regression'], ['x', 'f(x)'])
        plt.show()


    def createRLS(self):
        x = np.linspace(-4 * np.pi, 4 * np.pi, 500)

        test_size = len(x)
        lam = 0.95
        num_vars = 2
        LS = myRLS(num_vars, lam)
        pred_x = []
        pred_y = []
        X = np.matrix(np.zeros((1, num_vars)))
        pred_error = []
        for t in range(test_size - num_vars):
            for j in range(num_vars):
                X[0, j] = self.f(x[t + j])
            y1 = self.f(x[t + num_vars])  # predict
            pred_x.append(t)
            pred_y.append(float(X * LS.w))
            print("index, prediction, output", t + num_vars + 1, float(X * LS.w), float(y1))
            pred_error.append(LS.get_error())
            LS.add_obs(X.T, y1)
        #ax = plt.plot(pred_x[0:], pred_y[0:], label='predicted')
        #ax = plt.plot(pred_x[0:], self.f(x[num_vars:]), label='actual')
        ax = plt.plot(pred_x[200:], pred_error[200:], label='Error/residual')
        #_ = plt.plot(x[num_vars:], self.create()[num_vars:], label='simple linear regression')
        #plt.title("prediction for the function  sin(x) + 0.5 * x")

        plt.legend()
        plt.show()

            # sum += abs(df[i] - pred_y[i])







