import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from xgboost_function import CopyrightXGBoost


class TestCopyRightXGBoost:
    def setup_method(self):
        self.test_xgb = CopyrightXGBoost()
        self.test_xgb.load_existing(Path("TrainedModels/xgboost_sql_data_9_11.json"))
        with open("Test/test_data.json") as file:
            self.test_datapoint = json.load(file)

    def test_init(self):
        assert isinstance(self.test_xgb._xgb_model, xgb.XGBClassifier)
        assert isinstance(self.test_xgb._label_encoder, LabelEncoder)
        assert isinstance(self.test_xgb._openaccesscolor_encoder, LabelEncoder)

    def test_process_debug_column(self):
        cols_before = len(list(self.test_datapoint.keys()))
        test_datapoint = self.test_xgb._process_debug_column(self.test_datapoint)
        cols_after = len(list(test_datapoint.keys()))

        number_of_new_cols = 33
        assert cols_after - cols_before == number_of_new_cols

    def test_process_boolean_features(self):
        test_datapoint = self.test_xgb._process_debug_column(self.test_datapoint)
        test_datapoint = self.test_xgb._process_boolean_features(test_datapoint)

        for value in test_datapoint.values():
            assert value not in {
                "False",
                "'False'",
                "True",
                "'True'",
                '"False"',
                '"True"',
            }

    def test_process_openaccesscolor(self):
        test_datapoint = self.test_xgb._process_debug_column(self.test_datapoint)
        test_datapoint = self.test_xgb._process_boolean_features(test_datapoint)
        test_datapoint = self.test_xgb._process_openaccesscolor(test_datapoint)
        assert isinstance(test_datapoint["openaccesscolor"], int)

    def test_get_input_features(self):
        test_datapoint = self.test_xgb._process_debug_column(self.test_datapoint)
        test_datapoint = self.test_xgb._process_boolean_features(test_datapoint)
        test_datapoint = self.test_xgb._process_openaccesscolor(test_datapoint)

        input_features = self.test_xgb._get_input_features(test_datapoint)

        assert isinstance(input_features, np.ndarray)
        assert input_features.shape == (1, 33)
        assert input_features.dtype == float

    def test_process_data(self):
        input_features = self.test_xgb._process_datapoint(self.test_datapoint)

        assert isinstance(input_features, np.ndarray)
        assert input_features.shape == (1, 33)
        assert input_features.dtype == float

    def test_xgboost_prediction(self):
        model_output = self.test_xgb.xgboost_prediction(self.test_datapoint)

        assert model_output["prediction"] in self.test_xgb._label_encoder.classes_
        assert isinstance(model_output["prediction_probability"], float)
        assert 0 <= model_output["prediction_probability"] <= 1

    def test_read_data(self, monkeypatch):
        monkeypatch.setattr(self.test_xgb, "_fit_encoders", lambda *args: None)
        x_data, y_data = self.test_xgb._read_data(
            "./Test/test_data.csv", "this/does/not/matter"
        )

        assert x_data.shape == (2, 33)
        assert y_data.shape == (2,)

    def test_eval_model(self):
        test_out_path = Path("./Test/out")
        for child in test_out_path.glob("*"):
            print(child)
            if child.is_file():
                child.unlink()

        test_x_data = np.random.uniform(low=-0.5, high=2.5, size=(50, 33))
        test_y_data = np.random.randint(low=0, high=6, size=(50,))

        test_train_partition = 0.8

        self.test_xgb._eval_model(
            test_x_data, test_y_data, test_train_partition, test_out_path
        )

        assert Path("./Test/out/classification_report.json").exists()
