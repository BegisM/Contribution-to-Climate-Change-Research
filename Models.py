from getting_data import get_training_data, get_data, get_month_of_year, get_all_data, get_month_day, get_single_season_data, get_single_season_data_controlled_data
import numpy as np
from gradient_descent import gradient_descent
from normalization import normalize_back
import matplotlib.pyplot as plt
import pandas as pd
import dill

class PredictionModel:
    def __init__(self, raw_data=None):
        self.raw_data = raw_data
        self.X_normalized, self.X_mu, self.X_sigma = None, None, None
        self.Y_normalized, self.Y_mu, self.Y_sigma = None, None, None
        self.X, self.Y = None, None
        self.W, self.B = None, None
        self.J_h, self.p_h = None, None
        self.temp_pred, self.temp_pred_normalized = None, None
        self.first_april_temp_normalized, self.first_april_temp, self.first_april_reached_index = None, None, None

    def get_training_data(self, temp_type='TMAX'):
        (self.X_normalized, self.X_mu, self.X_sigma), (self.Y_normalized, self.Y_mu, self.Y_sigma) = get_training_data(self.raw_data, temp_type)
        self.X = normalize_back(self.X_normalized, self.X_mu, self.X_sigma)

    def train_model(self, alpha=0.06, iterations=100_000, old_model=None):
        self.get_training_data()
        if not old_model:
            w = np.random.randn(2, 1)
            b = 0.1
            self.W, self.B, self.J_h, self.p_h = gradient_descent(self.X_normalized, self.Y_normalized, w, b, alpha, iterations)
        else:
            self.W, self.B, self.J_h, self.p_h = gradient_descent(self.X_normalized, self.Y_normalized, old_model.W, old_model.B, alpha, iterations)

        self.compute_temperature()

        if self.X[-1, 0] >= 166:
            try:
                april_first_index = self.raw_data[self.raw_data["MONTH_DAY"] == "04-01"].index[0]
                self.first_april_temp_normalized = self.temp_pred_normalized[april_first_index]
                self.first_april_temp = self.temp_pred[april_first_index]
                self.first_april_reached_index = april_first_index
            except IndexError:
                self.first_april_temp_normalized = None
                self.first_april_temp = None
                self.first_april_reached_index = None

        return self.W, self.B

    def compute_temperature(self):
        self.temp_pred_normalized = self.W[0] * self.X_normalized[:, 0] + self.W[1] * self.X_normalized[:, 1] + self.B
        self.temp_pred = normalize_back(self.temp_pred_normalized, self.Y_mu, self.Y_sigma)



class PredictionMiniModels:
    def __init__(self, main_model=None, year=1990, start_day=None, start_month=None):
        if start_day and start_month:
            self.raw_all_data = get_single_season_data_controlled_data(year, start_month, start_day)
        else:
            self.raw_all_data = get_single_season_data(year)
        self.main_Model = main_model
        self.models = []
        self.starting_difference = self.raw_all_data.index.stop - self.main_Model.raw_data.index.stop if self.main_Model else None
        self.models_reached = []

    def train_models(self):
        if not self.starting_difference:
            return

        for i in range(0, 107):
            model = PredictionModel(raw_data=self.raw_all_data[:106+i+self.starting_difference])
            if i == 0:
                model.train_model()
            else:
                model.train_model(iterations=10_000, old_model=self.models[-1])

            self.models.append(model)
            self.compute_reached_model()

    def compute_reached_model(self):
        for i, temp in enumerate(self.models[-1].temp_pred[105+self.starting_difference:]):
            if temp >= self.main_Model.first_april_temp:
                self.models_reached.append({
                    'model': self.models[-1],
                    'temp': temp,
                    'difference': 182 - (i + 105)
                })
                self.models[-1].first_april_reached_index = self.starting_difference + i + 105
                self.models[-1].x_plot_end_reached = self.models[-1].first_april_reached_index if self.starting_difference > 0 else self.models[-1].first_april_reached_index - self.starting_difference
                return


class VisualizationOfModels:
    def __init__(self, main_model=None, mini_models=None):
        self.main_Model = main_model
        self.mini_Models = mini_models
        self.x_plot_dates = []
        self.months = []
        self.months_labels = []
        self.x_plot_values = []

        self.calculate_labels()

        self.first_april_index = 182 + self.mini_Models.starting_difference if self.mini_Models.starting_difference > 0 else 182
        self.tick_indices = [i for i, label in enumerate(self.x_plot_dates) if label in self.months]


    def calculate_labels(self):
        if not (self.mini_Models and self.main_Model):
            return

        x_plot = self.mini_Models.models[-1].X[:,0] if self.mini_Models.starting_difference > 0 else self.main_Model.X[:,0]

        raw_data = self.mini_Models.raw_all_data if self.mini_Models.starting_difference > 0 else self.main_Model.raw_data


        self.x_plot_dates = [f"{int(m):02d}-{int(d):02d}" for m, d in [(get_month_day(raw_data, round(i) - 1)) for i in x_plot]]
        self.x_plot_values = list(range(len(self.x_plot_dates)))
        self.months = ['10-01', '11-01', '12-01', '01-01', '02-01', '03-01', '04-01']
        self.months_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

        if self.mini_Models.starting_difference > 0:
            self.months.insert(0, f'09-{31-self.mini_Models.starting_difference}')
            self.months_labels.insert(0, 'Sep')

        self.main_Model.x_plot_start_index = self.mini_Models.starting_difference if self.mini_Models.starting_difference > 0 else 0
        self.main_Model.x_plot_end_index = self.main_Model.x_plot_start_index + round(self.main_Model.X[-1,0])

        for i in range(len(self.mini_Models.models)):
            self.mini_Models.models[i].x_plot_start_index = 0 if self.mini_Models.starting_difference > 0 else -self.mini_Models.starting_difference - 1
            self.mini_Models.models[i].x_plot_end_index = round(self.mini_Models.models[i].X[-1,0]) if self.mini_Models.starting_difference > 0 else round(self.mini_Models.models[i].X[-1,0]) - self.mini_Models.starting_difference



        return


    def plot_main_model(self):
        plt.figure(figsize=(12, 6))


        df_filtered_later = get_data(1950, 1980)
        plt.scatter(df_filtered_later['MONTH_DAY'], df_filtered_later['TMAX'], color='red', alpha=0.3, label='Min Temperature (TMAX)')

        difference = self.mini_Models.starting_difference if self.mini_Models.starting_difference > 0 else 0

        start_idx = self.main_Model.x_plot_start_index - difference
        end_idx = self.main_Model.x_plot_end_index - difference
        plt.scatter(self.x_plot_values[start_idx:end_idx], self.main_Model.temp_pred, color='blue', alpha=0.8, label='Main Model')


        plt.xlabel('Month')
        plt.ylabel('Temperature')
        indicator = 0
        if self.mini_Models.starting_difference > 0:
            indicator = 1
        plt.xticks([self.x_plot_dates[i] for i in self.tick_indices[indicator:]], self.months_labels[indicator:])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_models(self):
        plt.figure(figsize=(12, 6))

        for i, model in enumerate(self.mini_Models.models[:]):
            if i % 10 == 0 or i == 105:  # Keep your sampling logic
                start_idx = model.x_plot_start_index
                end_idx = model.x_plot_end_index
                plt.scatter(self.x_plot_values[start_idx:end_idx], model.temp_pred, color='green', alpha=0.8, label='Mini Model' if i == 0 else None)

        start_idx = self.main_Model.x_plot_start_index
        end_idx = self.main_Model.x_plot_end_index
        plt.scatter(self.x_plot_values[start_idx:end_idx], self.main_Model.temp_pred, color='blue', alpha=0.8, label='Main Model')

        plt.xticks(self.tick_indices, self.months_labels, rotation=45)
        plt.xlabel('Month')
        plt.ylabel('Temperature')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_models_reached(self):
        plt.figure(figsize=(12, 6))

        difference_days = self.mini_Models.starting_difference if self.mini_Models.starting_difference > 0 else 0

        for model_dict in self.mini_Models.models_reached[:]:
            start_idx = model_dict['model'].x_plot_start_index
            end_idx = model_dict['model'].x_plot_end_index
            plt.scatter(self.x_plot_values[start_idx:end_idx], model_dict['model'].temp_pred, color='green', alpha=0.01) #, label='Mini Model')
            # plt.axvline(x=model_dict['difference'], color='red', linestyle='--', label="April 1st")


        start_idx = self.main_Model.x_plot_start_index
        end_idx = self.main_Model.x_plot_end_index
        plt.scatter(self.x_plot_values[start_idx:end_idx], self.main_Model.temp_pred, color='blue', alpha=0.8, label='Main Model')
        plt.axvline(x=182+difference_days, color='red', linestyle='--', label="April 1st")

        plt.xticks(self.tick_indices, self.months_labels, rotation=45)
        plt.xlabel('Month')
        plt.ylabel('Temperature')
        plt.legend()
        plt.tight_layout()
        plt.show()


    def plot_one_model_reached(self, index=0):
        plt.figure(figsize=(12, 6))


        model_dict = self.mini_Models.models_reached[index]
        start_idx = model_dict['model'].x_plot_start_index



        end_idx = model_dict['model'].x_plot_end_reached
        first_april_reached = self.first_april_index - model_dict['difference']
        print(f"start index reached {start_idx}")
        print(f"end index reached {end_idx}")
        print(f"first april reached  {first_april_reached}")
        print(model_dict['model'].temp_pred[:end_idx].shape)
        plt.scatter(self.x_plot_values[start_idx:end_idx], model_dict['model'].temp_pred[:end_idx], color='green', alpha=0.7) #, label='Mini Model')


        plt.axvline(x=first_april_reached, color='blue', linestyle='--', label="April 1st")

        print(model_dict['difference'])

        start_idx = self.main_Model.x_plot_start_index
        end_idx = self.main_Model.x_plot_end_index
        plt.scatter(self.x_plot_values[start_idx:end_idx], self.main_Model.temp_pred, color='blue', alpha=0.8, label='Main Model')
        plt.axvline(x=self.first_april_index, color='red', linestyle='--', label="April 1st")

        plt.xticks(self.tick_indices, self.months_labels, rotation=45)
        plt.xlabel('Month')
        plt.ylabel('Temperature')
        plt.legend()
        plt.tight_layout()
        plt.show()



class MiniModelsDifferenceCheck:
    def __init__(self, models=None):
        if not models:
            return

        self.models = models
        self.models_differences = list()

        self.models_computation()

        self.models_differences = np.array(self.models_differences).reshape(-1, 1)
        self.linear_regression_model = None

        self.train_model()

    def models_computation(self, mean=False, first_reached=True, standard_deviation=False):
        for model in self.models:
            model_difference_list = list()
            for model_reached in model.models_reached:
                diff = model_reached.get('difference')
                model_difference_list.append(diff)
            model.models_difference = model_difference_list

            if first_reached:
                self.models_differences.append(model_difference_list[0])

            else:
                self.models_differences.append(model_difference_list[-1])

    def train_model(self):
        x = np.array([i for i in range(1, self.models_differences.shape[0] + 1)]).reshape(-1, 1)
        self.linear_regression_model = RegressionModel(x, self.models_differences)


    def plot_model(self):
        self.linear_regression_model.plot_model()


class RegressionModel:
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
        self.W, self.B, self.J_h, self.p_h = None, None, None, None
        self.train_model()
        self.prediction = self.compute_prediction()
        self.x_plot = np.array([1980 + i for i in range(0, self.X.shape[0])])

    def train_model(self, alpha=0.01, iterations=100_000):
        w = np.random.randn(1, 1)
        b = 0.1
        self.W, self.B, self.J_h, self.p_h = gradient_descent(self.X, self.Y, w, b, alpha, iterations)


    def compute_prediction(self):
        prediction = self.X.dot(self.W) + self.B
        return prediction

    def plot_model(self):
        plt.figure(figsize=(12, 6))

        plt.scatter(self.x_plot, self.Y, color='blue', label='Regression Model')
        plt.plot(self.x_plot, self.prediction, color='red', label='Prediction')

        # Create 10 integer ticks on x-axis
        x_ticks = np.linspace(self.x_plot.min(), self.x_plot.max(), 21, dtype=int)
        plt.xticks(ticks=x_ticks, labels=x_ticks, rotation=0)

        plt.xlabel('Year')
        plt.ylabel('Days')
        plt.legend()
        plt.tight_layout()
        plt.show()
