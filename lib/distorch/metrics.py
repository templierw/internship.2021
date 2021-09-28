from sklearn.metrics import mean_squared_error, mean_absolute_error

class PerformanceEvaluator():

    def __init__(
        self,
        y_pred,
        y_true,
        scale
    ) -> None:

        self.y_pred = y_pred
        self.y_true = y_true
        self.scale = scale

        self.metrics = {
            'rmse': 0.0,
            'mape': 0.0
        }
        
    def rmse(self, perc=True):
        return self.__metric_fn(self.__rmse, perc=perc)

    def mape(self):
        return self.__metric_fn(mean_absolute_error, perc=True)


    def __metric_fn(self, fn, perc=False):
        rmetric = fn(self.y_pred, self.y_true)/self.scale

        return round(rmetric * 100 if perc else rmetric, 3)
    
    @staticmethod
    def __rmse(y_pred, y_true):
        return mean_squared_error(y_pred, y_true)**0.5

    def get_performance_metrics(self):

        self.metrics['rmse'] = self.rmse()
        self.metrics['mape'] = self.mape()

        return self.metrics