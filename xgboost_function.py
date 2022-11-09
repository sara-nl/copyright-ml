import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

encoder_openaccesscolor = LabelEncoder()
encoder_openaccesscolor.classes_ = np.load(
    "FittedEncoders/encoder_openaccess_color_classes.npy"
)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("FittedEncoders/label_encoder_classes.npy")

xgb_model = xgb.XGBClassifier()
xgb_model = xgb.load("TrainedModels/xgboost_sql_data_9_11.json")


def get_correct_cols(extra_cols: set) -> list:
    needed_cols = {
        "id",
        "uuid",
        "url",
        "filesource",
        "filestatus",
        "filemimetype",
        "filename",
        "filehash",
        "filedate",
        "lastmodifieddate",
        "isfilepublished",
        "wordcount",
        "pagecount",
        "filescanresults",
        "doi",
        "issn",
        "isbn",
        "author",
        "title",
        "publisher",
        "publicationyear",
        "sourcepagecount",
        "sourcewordcount",
        "usedpages",
        "filetype",
        "userexcludedforscan",
        "usedmultiplesources",
        "isopenaccesstitle",
        "openaccesslink",
        "apacitation",
        "contactcomment",
        "skippedcomment",
        "finaladvice",
        "creator",
        "debug",
        "depublicationdate",
        "document",
        "filepath",
        "filesize",
        "first_scan_result",
        "first_scan_result_date",
        "human_intervention",
        "incollection",
        "institution_intervention_result",
        "institution_intervention_result_date",
        "last_scan_result",
        "last_scan_result_date",
        "oclcnumber",
        "openaccesscolor",
        "picturecount",
        "prediction",
        "predictionmatrix",
        "publicationdate",
        "recordlastmodified",
        "reliability",
        "rescan",
        "runidentifier",
        "course_id",
        "jstor",
        "always",
        "doi_in_oa",
        "name",
        "manual_classification",
        "remarks",
    }

    if needed_cols == extra_cols or needed_cols.issubset(extra_cols):
        return list(needed_cols)

    raise ValueError(
        "Not all needed keys are present in this datapoint's 'debug' column"
    )


def process_debug_column(datapoint: dict) -> dict:
    debug_data = json.loads(datapoint["debug"])
    extra_cols = get_correct_cols(set(debug_data.keys()))
    for col in extra_cols:
        datapoint[col] = debug_data[col]
    return datapoint


def encode_openaccesscolor_to_int(datapoint: dict) -> dict:
    datapoint["openaccesscolor"] = encoder_openaccesscolor.transform(
        datapoint["openaccesscolor"]
    )
    return datapoint


def transform_boolean_features(datapoint: dict) -> dict:
    items = datapoint.items()
    for key, value in items:
        if value in {False, "False", "'False'"} or value is None:
            datapoint[key] = 0
        elif value in {True, "True", "'True'"}:
            datapoint[key] = 1
    return datapoint


def get_input_features(datapoint: dict) -> np.ndarray:
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
        "manual_classification",
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
    return np.asarray(values)


def preprocess_data(datapoint: dict) -> np.ndarray:
    datapoint = process_debug_column(datapoint)
    datapoint = transform_boolean_features(datapoint)
    datapoint = encode_openaccesscolor_to_int(datapoint)

    input_features = get_input_features(datapoint)
    return input_features


def xgboost_prediction(datapoint: dict) -> dict:
    input_features = get_input_features(datapoint)
    prediction = xgb_model.predict(input_features)
    prediction_proba = xgb_model.predict_proba(input_features)
