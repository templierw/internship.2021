import matplotlib.pyplot as plt
import os

def plot_prob_forecasts(
    ts_entry,
    forecast_entry,
    prediction_length: int,
    prediction_intervals: list = (95.0,),
    save: bool = False,
    name: str = "plot"
) -> None:

    legend = (["observations", "median prediction"] + 
              [f"{k}% prediction interval" for k in prediction_intervals][::-1]
            )

    _, ax = plt.subplots(1, 1, figsize=(30, 7))
    ts_entry[-prediction_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")

    plt.savefig(f"{os.environ['VIZ_DIR']}/{name}.png") if save else plt.show()
    plt.clf()