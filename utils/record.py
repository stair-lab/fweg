import csv
import os
import json

from typing import Union, List


def format_fweg_extra(
    num_iters: int,
    groups: Union[List[str], int],
    groups_descr: str,
    add_all: bool,
    epsilon: float,
    FW_val_flag: bool,
) -> str:
    data = {
        "groups": groups,
        "groups_descr": groups_descr,
        "add_all": add_all,
        "epsilon": epsilon,
        "FW_val_flag": FW_val_flag,
    }
    return json.dumps(data)


class Results_Recorder:
    columns = [
        "Dataset",
        "Seed",
        "Model_Type",
        "Metric",
        "Val",
        "Hyper_Val",
        "Test",
        "Extra",
    ]

    def __init__(self, savepath: str, dataset: str):
        self.dataset = dataset
        self.fp = None
        self.writer = None
        if os.path.exists(savepath):
            print("Results file exists, appending to it...")
            self.fp = open(savepath, mode="a")
            self.writer = csv.writer(self.fp)
        else:
            print("Results file does not exist, creating it...")
            self.fp = open(savepath, mode="w")
            self.writer = csv.writer(self.fp)
            self.writer.writerow(Results_Recorder.columns)

    def save(
        self,
        seed: int,
        metric: str,
        model_type: str,
        val_score: float,
        hyper_val_score: float,
        test_score: float,
        extra: str,
    ):
        self.writer.writerow(
            [
                self.dataset,
                seed,
                model_type,
                metric,
                val_score,
                hyper_val_score,
                test_score,
                extra,
            ]
        )

    def close(self):
        self.fp.close()


class Mock_Recorder:
    def __init__(self):
        pass

    def save(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass
