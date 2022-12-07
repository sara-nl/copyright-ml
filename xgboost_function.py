import json
from math import isnan

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


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
        if isnan(datapoint["openaccesscolor"]):
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
        values = [datapoint[feature] for feature in input_features]

        return np.asarray(values).reshape((1, -1))

    def _process_datapoint(self, datapoint: dict) -> np.ndarray:
        datapoint = self._process_debug_column(datapoint)
        datapoint = self._process_boolean_features(datapoint)
        datapoint = self._process_openaccesscolor(datapoint)

        input_features = self._get_input_features(datapoint)
        return input_features

    def xgboost_prediction(self, datapoint: dict) -> dict:
        input_features = self._process_datapoint(datapoint)
        prediction = self._xgb_model.predict(input_features)
        prediction_proba = self._xgb_model.predict_proba(input_features)[0][prediction]

        output = dict()
        output["prediction"] = self._label_encoder.inverse_transform(prediction)[0]
        output["prediction_probability"] = float(prediction_proba[0])

        return output
