import csv
import json
from ast import literal_eval
from math import isnan

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class CopyrightXGBoost:
    def __init__(self) -> None:
        self._xgb_model = xgb.XGBClassifier()
        self._label_encoder = LabelEncoder()
        self._openaccesscolor_encoder = LabelEncoder()

    def load_existing(self, model_filepath):
        self._xgb_model.load_model(model_filepath)
        self._label_encoder.classes_ = np.load(
            "FittedEncoders/label_encoder_classes.npy", allow_pickle=True
        )
        self._openaccesscolor_encoder.classes_ = np.load(
            "FittedEncoders/openaccescolor_encoder_classes.npy", allow_pickle=True
        )

    def _fit_encoders(self, data, out_path):
        labels = [datapoint["manual_classification"] for datapoint in data]
        self._label_encoder.fit(labels)
        np.save(
            out_path / "label_encoder_classes.npy",
            self._label_encoder.classes_,
        )

        # TODO: get all colors and fit and save new encoder
        self._openaccesscolor_encoder.classes_ = np.load(
            "FittedEncoders/openaccescolor_encoder_classes.npy", allow_pickle=True
        )
        np.save(
            out_path / "openaccescolor_encoder_classes.npy",
            self._openaccesscolor_encoder.classes_,
        )

    def _get_cols(self, extra_cols: set) -> list:
        needed_cols = {
            "jstor",
            "always",
            "DOI_in_OA",
            "DOI_no_PPT",
            "PPT_in_name",
            "ppt_creator",
            "wordcount_o",
            "10_pics_page",
            "Contains_DOI",
            "Contains_ISBN",
            "creator_abbyy",
            "words_page>350",
            "doc in metadata",
            "keyword_creator",
            "ppt in metadata",
            "Creative commons",
            "Words_more_300pp",
            "file_ext_mp3_wav",
            "file_ext_mp4_mov",
            "10>_Pagecount_<50",
            "Contains_copyright",
            "Kleiner_10_paginas",
            "filename_indicator",
            "Contains_sciencemag",
            "Pagecount_bigger_50",
            "book_and_words<10000",
            "Contains_published_in",
            "Contains_researchgate",
            "Contains_to_appear_in",
            "Is_journal_words<8000",
            "images_same_pagecount",
            "xls in metadata titel",
            "Publisher_from_crossref",
            "Contains_recommended_citation",
            "Minder dan 50 woorden per pagina",
        }

        if needed_cols == extra_cols or needed_cols.issubset(extra_cols):
            return list(needed_cols)

        raise ValueError(
            "Not all needed keys are present in this datapoint's 'debug' column"
        )

    def _process_debug_column(self, datapoint: dict) -> dict:
        debug_data = json.loads(datapoint["debug"])
        extra_cols = self._get_cols(set(debug_data.keys()))
        for col in extra_cols:
            datapoint[col] = debug_data[col]
        return datapoint

    def _process_boolean_features(self, datapoint: dict) -> dict:
        items = datapoint.items()
        for key, value in items:
            if value in {False, "False", "'False'", '"False"'} or value is None:
                datapoint[key] = 0
            elif value in {True, "True", "'True'", '"True"'}:
                datapoint[key] = 1
        return datapoint

    def _process_openaccesscolor(self, datapoint: dict) -> dict:
        if datapoint["openaccesscolor"] == "" or isnan(datapoint["openaccesscolor"]):
            datapoint["openaccesscolor"] = "nan"
        datapoint["openaccesscolor"] = int(
            self._openaccesscolor_encoder.transform(
                np.array([datapoint["openaccesscolor"]])
            )[0]
        )
        return datapoint

    def _get_input_features(self, datapoint: dict) -> np.ndarray:
        input_features = [
            "filestatus",
            "isfilepublished",
            "wordcount",
            "pagecount",
            "isopenaccesstitle",
            "filesize",
            "incollection",
            "openaccesscolor",
            "picturecount",
            "reliability",
            "doi_in_oa",
            "DOI_in_OA",
            "DOI_no_PPT",
            "PPT_in_name",
            "ppt_creator",
            "wordcount_o",
            "10_pics_page",
            "Contains_DOI",
            "Contains_ISBN",
            "creator_abbyy",
            "words_page>350",
            "keyword_creator",
            "Creative commons",
            "Words_more_300pp",
            "10>_Pagecount_<50",
            "Contains_copyright",
            "Kleiner_10_paginas",
            "filename_indicator",
            "Pagecount_bigger_50",
            "book_and_words<10000",
            "Contains_published_in",
            "images_same_pagecount",
            "Minder dan 50 woorden per pagina",
        ]

        values = [
            literal_eval(datapoint[feature])
            if (type(datapoint[feature]) == str)
            else datapoint[feature]
            for feature in input_features
        ]

        return np.asarray(values).reshape((1, -1))

    def _process_datapoint(self, datapoint: dict) -> np.ndarray:
        datapoint = self._process_debug_column(datapoint)
        datapoint = self._process_boolean_features(datapoint)
        datapoint = self._process_openaccesscolor(datapoint)

        input_features = self._get_input_features(datapoint)
        return input_features

    def _read_data(self, data_filepath, out_path):
        with open(data_filepath) as file:
            reader = csv.DictReader(file)
            data = list(reader)
        self._fit_encoders(data, out_path)
        x_data = np.asarray(
            [self._process_datapoint(datapoint).squeeze() for datapoint in data]
        )
        y_data = self._label_encoder.transform(
            [datapoint["manual_classification"] for datapoint in data]
        )
        return x_data, y_data

    def _eval_model(self, x_data, y_data, train_partition, out_path):
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, train_size=train_partition
        )

        self._xgb_model.fit(x_train, y_train)

        predictions = self._xgb_model.predict(x_test)
        report = classification_report(
            y_test,
            predictions,
            target_names=self._label_encoder.classes_,
            labels=np.unique(y_data),
            output_dict=True,
            zero_division=0,
        )

        print(type(out_path))
        print(out_path)

        with open(
            out_path / "classification_report.json", "w", encoding="utf-8"
        ) as file:
            json.dump(report, file, ensure_ascii=False)

    def train_xgboost(self, data_filepath, out_path, train_partition):
        x_data, y_data = self._read_data(data_filepath, out_path)
        self._eval_model(x_data, y_data, train_partition, out_path)
        self._xgb_model.fit(x_data, y_data)
        self._xgb_model.save_model(out_path / "xgboost.json")

    def xgboost_prediction(self, datapoint: dict) -> dict:
        input_features = self._process_datapoint(datapoint)
        prediction = self._xgb_model.predict(input_features)
        prediction_proba = self._xgb_model.predict_proba(input_features)[0][prediction]

        output = dict()
        output["prediction"] = self._label_encoder.inverse_transform(prediction)[0]
        output["prediction_probability"] = float(prediction_proba[0])

        return output
