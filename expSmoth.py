import pandas as pd
import matplotlib.pyplot as plt


class ExpSmoothing:

    def __init__(self, alpha):
        self.alpha = alpha

    def expSmoothing(self):
        df_ice_cream = pd.read_csv('ice_cream.csv')
        df_ice_cream.rename(columns={'DATE': 'date', 'IPN31152N': 'production'}, inplace=True)
        df_ice_cream['date'] = pd.to_datetime(df_ice_cream.date)
        df_ice_cream.set_index('date', inplace=True)
        start_date = pd.to_datetime('2010-01-01')
        df_ice_cream = df_ice_cream[start_date:]
        y = pd.DataFrame(df_ice_cream)
        print(y)
        df = y['production'].values
        # print(df)
        test_size = len(df)
        pred_x = []
        pred_y = []
        pred_error = []
        sum = 0
        sumq = 0
        for i in range(test_size):
            pred_x.append(i)
            if i == 0:
                pred_y.append(0)
            else:
                pred_y.append(pred_y[i - 1] + self.alpha * (df[i - 1] - pred_y[i - 1]))
            sum += abs(df[i] - pred_y[i])
            sumq += (df[i] - pred_y[i]) ** 2

            pred_error.append(df[i] - pred_y[i])

        ax = plt.plot(pred_x[0:], pred_y[0:], label='predicted')
        _ = plt.plot(pred_x[0:], df[0:], label='actual')
        _ = plt.plot(pred_x[0:], pred_error[0:], label='Error/residual')

        plt.title("ice cream production prediction")
        plt.text(50, 50, 'alpha=0.5')

        #########################################################
        plt.legend()
        plt.show()
        print(sum / len(df))
        print(sumq / len(df))

        for i in range(test_size):
            print("actual, predicted, Error", df[i], pred_y[i], pred_error[i])
