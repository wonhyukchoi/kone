import os
import time
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from pprint import pprint
from .models import Kone


class KoneTrainer:
    def __init__(self, train_data: pd.DataFrame,
                 seed=0x2f):
        np.random.seed(seed)
        self._x, self._y = self._to_x_and_y(train_data)

    def train_batch(self, hyperparameters,
                    save_path: str,
                    acc_metric='val_categorical_accuracy',
                    result_file='results.csv'):

        for single_hyperparameter in hyperparameters:
            hparam_name = ""
            for key, value in single_hyperparameter.items():
                hparam_name += key[:2] + str(value)

            save_name = os.path.join(save_path, hparam_name)

            before = time.time()
            train_acc = self.train(hyperparameter=single_hyperparameter,
                                   save_name=save_name,
                                   acc_metric=acc_metric, plot=False)
            elapsed = time.time() - before

            elapsed_mins = int(elapsed / 60)
            now = datetime.now().strftime("%Y-%m-%d T%H:%M:%S")

            results = {'time': now, 'accuracy': train_acc,
                       'train_time(min)': elapsed_mins,
                       'hyperparameter': hparam_name}

            with open(result_file, 'a', encoding='utf-8') as f:
                csv_file = csv.DictWriter(f=f, fieldnames=results.keys())
                csv_file.writerow(results)
            pprint(results)

    def train(self, hyperparameter: dict, save_name: str,
              acc_metric='val_categorical_accuracy',
              plot=False, save=False):
        window_size = hyperparameter['window_size']
        epochs = hyperparameter['epochs']
        batch_size = hyperparameter['batch_size']
        embedding_dim = hyperparameter['embedding_dim']
        num_neurons = hyperparameter['num_neurons']
        optimizer = hyperparameter['optimizer']
        verbose = hyperparameter['verbose']

        kone = Kone(window_size=window_size)
        kone.train(x=self._x, y=self._y,
                   epochs=epochs, batch_size=batch_size,
                   embedding_dim=embedding_dim, num_neurons=num_neurons,
                   optimizer=optimizer, verbose=verbose)

        if save:
            index_name = save_name + '_index.json'
            weight_name = save_name + '_weight.h5'
            kone.save_model(index_name=index_name, weight_name=weight_name)

        if plot:
            kone.plot_train_history(save_name=save_name)

        accuracy = round(max(kone.train_history[acc_metric]) * 100, 2)
        return accuracy

    def _to_x_and_y(self, data: pd.DataFrame) -> (list, list):
        raise NotImplementedError


if __name__ == "__main__":
    """ Hardcoded training """

    mode = 'single'

    result_path = os.path.join(os.getcwd(), 'train_results')
    save_path_name = os.path.join(os.path.join(result_path, 'train_models'), '_')
    csv_path = os.path.join(result_path, 'train_results.csv')

    data_path = os.path.join(os.path.join(os.path.join(
        os.getcwd(), os.pardir), 'data'), 'train_data.csv')
    kone_trainer = KoneTrainer(train_data=pd.read_csv(data_path))

    # TODO
    hyperparameter_list = [{}]

    if mode == 'batch':
        kone_trainer.train_batch(hyperparameters=hyperparameter_list,
                                 save_path=save_path_name)
    elif mode == 'single':
        model_save_path = os.path.join(os.path.join(os.getcwd(), 'models'), 'model_')
        kone_trainer.train(hyperparameter=hyperparameter_list[0],
                           save_name=model_save_path, plot=True, save=True)

    raise NotImplementedError
