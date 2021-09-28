from typing import Iterator, Tuple
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import Forecast
from pandas.core.series import Series
from pandas.core.frame import DataFrame

class Forecast:

    def __init__(
        self,
        predictor: Predictor
    ) -> None:
        self.__predictor = predictor
        self.__forecasted = False

    def evaluate_predictions(
        self,
        dataset: ListDataset,
        num_samples: int = 100
    ) -> Tuple[Iterator[Forecast], Iterator[Series]]:

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,  # test dataset
            predictor=self.__predictor,  # predictor
            num_samples=num_samples,  # number of sample paths we want for evaluation
        )

        f = list(forecast_it)
        t = list(ts_it)

        self.__forecasted = True
        self.__forecasts = f[0]
        self.__tss = t[0]

        return f, t

    def get_forecast_info(self) -> None:

        if not self.__forecasted:
            print('No forecast made yet')

        else:
            print(
                f"Start of the forecast window: {self.__forecasts.start_date}"
                )

    def get_forecast_entry(
        self
    ) -> Tuple[list, list]:

        return self.__forecasts, self.__tss

def compute_metrics(
    quantiles: list,
    tss: Series,
    forecasts: Forecast,
    num_series: int
) -> DataFrame:

    evaluator = Evaluator(quantiles=quantiles)

    agg, _ = evaluator(iter(tss), iter(forecasts), num_series=num_series)

    res = DataFrame(
        index = ['MSE', 'MASE', 'MAPE', 'RMSE', 'NRMSE'] + 
                [f'QuantileLoss[{q}]' for q in quantiles] + 
                [f'Coverage[{q}]' for q in quantiles] +
                [f'AVE[{q}]' for q in quantiles] +
                [f'wQuantileLoss[{q}]' for q in quantiles],
        columns = ['score']
        )

    for i in res.index:
        if i[0] != 'A':
            res.loc[i, 'score'] = agg[i]

    for q in quantiles:
        res.loc[f'AVE[{q}]', 'score'] = res.loc[f'Coverage[{q}]', 'score'] - q

    res.round(3)

    return res