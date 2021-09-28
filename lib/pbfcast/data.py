import pandas as pd
from pandas import DataFrame
import json
import os

from gluonts.dataset.common import ListDataset

class Dataset:

    def __init__(self, pv=True) -> None:
        self.__df = pd.read_csv(f"{os.environ['DATA_DIR']}/case03/case03_01.csv", index_col=0, parse_dates=True)

        self.__pv = pv

    def get_trainer_set(self) -> DataFrame:
        return self.__df

    def get_listdataset(
        self,
        df: DataFrame = None,
        freq: str = '1H'
    ) -> ListDataset:

        if df is None:
            df = self.__df

        features = [
            df['temp'],
            df['irrad'],
            df['Winter'],
            df['Spring'],
            df['Summer'],
            df['Autumn'],
            
        ] + [
            df[f'Hour{h}'] for h in range(0,24)
        ]

        if self.__pv:
            features + [
                df['irrad']
            ]

        return ListDataset(
            data_iter=[{
                'start': df.index[0],
                'target': df.pv if self.__pv else df.load,
                'feat_dynamic_real': features
            }],
            freq=freq,
            one_dim_target=True
        )

    def getSubset(
        self,
        months: list,
        df: DataFrame = None
    ) -> DataFrame:

        if df is None:
            df = self.__df

        return df[df.index.month.isin(months)]

class MonthXP:

    def __init__(self) -> None:
        import os
        self.xp_file = open(f"{os.environ['ROOT_DIR']}/notebooks/pbfcast_cases/months.json",)
        self.data = json.load(self.xp_file)

        self.months_int = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12
        }

        self.nb_XP = len(self.data)

    def __del__(self) -> None:
        self.xp_file.close()

    def get_months_list(
        self,
        xp: int
    ) -> list:

        xp = self.data[f'xp{xp}']

        return [
            value for key, value in self.months_int.items() if key in xp
        ]