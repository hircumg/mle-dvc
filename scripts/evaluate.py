# scripts/evaluate.py
import json
import os

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, cross_validate


# оценка качества модели
def evaluate_model():
    # прочитайте файл с гиперпараметрами params.yaml
    with open("params.yaml") as fd:
        params = yaml.safe_load(fd)

    # загрузите результат прошлого шага: fitted_model.pkl
    data = pd.read_csv("data/initial_data.csv", index_col=params["index_col"])
    with open("models/fitted_model.pkl", "rb") as fd:
        model = joblib.load(fd)

    # реализуйте основную логику шага с использованием прочтённых гиперпараметров
    # Проверка качества на кросс-валидации
    cv_strategy = StratifiedKFold(n_splits=params["n_splits"])
    cv_res = cross_validate(
        model,
        data,
        data[params["target_col"]],
        cv=cv_strategy,
        n_jobs=params["n_jobs"],
        scoring=params["metrics"],
    )

    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3)

    # сохраните результата кросс-валидации в cv_res.json
    os.makedirs("cv_results", exist_ok=True)
    with open("cv_results/cv_res.json", "w") as fp:
        json.dump(cv_res, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    evaluate_model()
