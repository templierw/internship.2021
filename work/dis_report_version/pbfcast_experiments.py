from collections import OrderedDict
from typing import Tuple
from pandas.core.frame import DataFrame

import argparse
import os

from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer

from pbfcast.data import Dataset, MonthXP
from pbfcast.hyperparameters import generate_parameter_space
from pbfcast.evaluate import Forecast, compute_metrics
from pbfcast.visualization import plot_prob_forecasts
from pbfcast.persistency import save_model

def get_datasets(
    w_params: OrderedDict,
    pv: bool
) -> Tuple[ListDataset, ListDataset, ListDataset]:

    m3_ds = Dataset(pv=pv)

    train_set = m3_ds.get_listdataset(
        df = m3_ds.getSubset(
                months=w_params['train_months']
        )
    )

    val_set = m3_ds.get_listdataset(
        df=m3_ds.getSubset(
            months=w_params['val_months'],
        )
    )

    hb_set = m3_ds.get_listdataset(
        df=m3_ds.getSubset(
            months=w_params['hb_months'],
        )
    )

    return train_set, val_set, hb_set

#--------------------------------------------------#


def train(
    train_set: ListDataset,
    w_params: OrderedDict,
    case
) -> Predictor:

    print(f'training with case: {case}')

    estimator = DeepAREstimator(
        freq=w_params['freq'],
        prediction_length=w_params['pred_length'],
        trainer=Trainer(
            epochs=case.epoch,
            learning_rate=case.lr
        ),
        use_feat_dynamic_real=True,
        context_length=case.context_length,
        scaling=True,
        num_layers=2,
        num_cells=case.nb_cells
    )

    return estimator.train(train_set)


def run(
    m_params: OrderedDict,
    pv: bool
) -> DataFrame:

    months_xp = MonthXP()

    for xp in range(1, months_xp.nb_XP + 1):

        print(f'training on months: {months_xp.get_months_list(xp)}')

        w_params = OrderedDict({
            'freq': '1H',
            'pred_length': 14*24,
            'train_months': months_xp.get_months_list(xp),
            'val_months': [5],
            'hb_months': [9],
        })

        train_set, val_set, _ = get_datasets(w_params, pv)

        cases = generate_parameter_space(m_params)

        for case in cases:
            name = f'deepAR_{("pv" if pv else "load")}_e{case.epoch}_cl{case.context_length}_nbc{case.nb_cells}_lr{case.lr}_xp{xp}'
            predictor = train(train_set, w_params, case)
            save_model(predictor, name)
            
            print(f'forecasting with case: {case}')

            f = Forecast(predictor)
            forecasts, tss = f.evaluate_predictions(val_set)
            f_entry, ts_entry = f.get_forecast_entry()

            plot_prob_forecasts(
                ts_entry,
                f_entry,
                prediction_length=w_params['pred_length'],
                prediction_intervals=[90.,],
                save=True,
                name=name
               )
            
            print(f'computing metrics with case: {case}')

            res = compute_metrics(
                quantiles=[.9,],
                tss=tss,
                forecasts=forecasts,
                num_series=len(val_set)
            )

            res.to_csv(f"{os.environ['RES_DIR']}/pbfcast/{name}.csv")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="PV or Load model",
                        type=str, choices=['pv', 'load'])
    args = parser.parse_args()

    pv = (args.model == 'pv')

    model_params = OrderedDict({
        'nb_cells': [30],
        'epoch': [1],
        'context_length': [24*7],
        'lr': [.005]
    })

    run(model_params, pv)


if __name__ == '__main__':
    main()
