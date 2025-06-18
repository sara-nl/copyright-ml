import csv
import json
from ast import literal_eval
from math import isnan
import pandas as pd 

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FEATURES = [
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
    # "doi_in_oa", # TODO: duplicate wiht the one below?
    "DOI_in_OA",
    "DOI_no_PPT",
    "PPT_in_name",
    "ppt_creator",
    "wordcount_o",
    "_10_pics_page",
    "Contains_DOI",
    "Contains_ISBN",
    "creator_abbyy",
    "words_page350",
    "keyword_creator",
    "Creativecommons",
    "Words_more_300pp",
    # "10>_Pagecount_<50", # TODO: goneee
    "Contains_copyright",
    "Kleiner_10_paginas",
    "filename_indicator",
    # "Pagecount_bigger_50", # TODO: goneee
    "book_and_words10000",
    "Contains_published_in",
    "images_same_pagecount",
    "Minderdan50woordenperpagina",
]
class CopyrightXGBoost:
    def __init__(self, random_state=42) -> None:
        self._xgb_model = xgb.XGBClassifier(random_state=random_state)
        self._label_encoder = LabelEncoder()
        self._openaccesscolor_encoder = LabelEncoder()
        self._random_state = random_state

    def load_existing(self, model_filepath):
        self._xgb_model.load_model(model_filepath)
        self._label_encoder.classes_ = np.load(
            "FittedEncoders/label_encoder_classes.npy", allow_pickle=True
        )
        self._openaccesscolor_encoder.classes_ = np.load(
            "FittedEncoders/openaccescolor_encoder_classes.npy", allow_pickle=True
        )

    def _fit_encoders(self, data, out_path):
        labels = [datapoint["V2 - Licentie Nodig"] for datapoint in data]
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
            "_10_pics_page",
            "Contains_DOI",
            "Contains_ISBN",
            "creator_abbyy",
            "words_page350",
            "doc in metadata",
            "keyword_creator",
            "ppt in metadata",
            "Creativecommons",
            "Words_more_300pp",
            "file_ext_mp3_wav",
            "file_ext_mp4_mov",
            # "10>_Pagecount_<50",
            "Contains_copyright",
            "Kleiner_10_paginas",
            "filename_indicator",
            "Contains_sciencemag",
            # "Pagecount_bigger_50",
            "book_and_words10000",
            "Contains_published_in",
            "Contains_researchgate",
            "Contains_to_appear_in",
            "Is_journal_words<8000",
            "images_same_pagecount",
            "xls in metadata titel",
            "Publisher_from_crossref",
            "Contains_recommended_citation",
            "Minderdan50woordenperpagina",
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
        if datapoint["openaccesscolor"] == "" or pd.isna(datapoint["openaccesscolor"]):
            datapoint["openaccesscolor"] = "nan"

        datapoint["openaccesscolor"] = int(
            self._openaccesscolor_encoder.transform(
                np.array([datapoint["openaccesscolor"]])
            )[0]
        )
        return datapoint

    def _get_input_features(self, datapoint: dict) -> np.ndarray:
        values = [
            literal_eval(datapoint[feature])
            if (type(datapoint[feature]) == str and datapoint[feature] != "")
            else datapoint[feature]
            for feature in INPUT_FEATURES
        ]

        return np.asarray(values).reshape((1, -1))

    def _process_datapoint(self, datapoint: dict) -> np.ndarray:
        # datapoint = self._process_debug_column(datapoint)
        datapoint = self._process_boolean_features(datapoint)
        datapoint = self._process_openaccesscolor(datapoint)

        input_features = self._get_input_features(datapoint)
        return input_features

    def _check_if_datapoint_has_data(self, datapoint: dict) -> bool:
        any_input = False
        for feature in INPUT_FEATURES:
            if feature in ["filestatus", "isfilepublished", "wordcount", "pagecount", "isopenaccesstitle", "incollection", "picturecount", "reliability", "openaccesscolor", "filesize"]:
                continue
            if feature in datapoint and (
                datapoint[feature] is not None
                and datapoint[feature] != ""
                and not (isinstance(datapoint[feature], str) and datapoint[feature].isspace())
            ):
                any_input = True
                break

        return any_input

    def _read_data(self, data_filepath, out_path):
        with open(data_filepath) as file:
            reader = csv.DictReader(file, delimiter="\t")
            data = list(reader)
        # filter out the warnings here such that it does not become a label
        data = [datapoint for datapoint in data if not datapoint["V2 - Licentie Nodig"] == "WARNING"]
        self._fit_encoders(data, out_path)

        x_data = []
        y_data = []
        lengte_data = []  # Store the "V2 - Lengte" column values
        skipped = 0
        for datapoint in data:
            if self._check_if_datapoint_has_data(datapoint):
                x_data.append(self._process_datapoint(datapoint))
                y_data.append(self._label_encoder.transform(
                    [datapoint["V2 - Licentie Nodig"]]
                )[0])  # Extract the scalar value from the array
                lengte_data.append(datapoint.get("V2 - Lengte", "Unknown"))  # Store lengte value
            else:
                skipped += 1
        
        print(f"Skipped {skipped} datapoints without data, remaining {len(x_data)}")

        # x_data = np.asarray(
        #     [self._process_datapoint(datapoint).squeeze() for datapoint in data]
        # )
        # y_data = self._label_encoder.transform(
        #     [datapoint["V2 - Licentie Nodig"] for datapoint in data]
        # )
        return x_data, y_data, lengte_data

    def _eval_model(self, x_data, y_data, lengte_data, train_partition, out_path, use_class_weights=False):
        # Ensure y_data is a proper numpy array
        y_data = np.asarray(y_data)
        lengte_data = np.asarray(lengte_data)
        
        x_train, x_test, y_train, y_test, lengte_train, lengte_test = train_test_split(
            x_data, y_data, lengte_data, train_size=train_partition, random_state=self._random_state
        )

        # Compute sample weights if requested
        sample_weight = None
        if use_class_weights:
            sample_weight = compute_sample_weight('balanced', y_train)
            print(f"Using class weights. Sample weight distribution: {np.bincount(y_train)}")

        self._xgb_model.fit(x_train, y_train, sample_weight=sample_weight)

        predictions = self._xgb_model.predict(x_test)
        report = classification_report(
            y_test,
            predictions,
            target_names=self._label_encoder.classes_,
            labels=np.unique(y_data),
            output_dict=True,
            zero_division=0,
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=np.unique(y_data))
        class_names = [self._label_encoder.classes_[i] for i in np.unique(y_data)]

        print("out path:", out_path)

        # Create formatted report
        self._write_formatted_report(report, cm, class_names, out_path)
        
        # Save confusion matrix as image
        self._create_confusion_matrix_plots(cm, class_names, out_path)
        
        # Generate subgroup confusion matrices by "V2 - Lengte"
        self._save_subgroup_confusion_matrices(y_test, predictions, lengte_test, class_names, out_path)

    def train_xgboost(self, data_filepath, out_path, train_partition, use_class_weights=False):
        self._xgb_model = xgb.XGBClassifier(random_state=self._random_state)
        x_data, y_data, lengte_data = self._read_data(data_filepath, out_path)
        x_data = np.asarray(x_data, dtype=object).squeeze()
        y_data = np.asarray(y_data)  # Ensure y_data is a proper numpy array
        lengte_data = np.asarray(lengte_data)  # Ensure lengte_data is a proper numpy array
        self._eval_model(x_data, y_data, lengte_data, train_partition, out_path, use_class_weights)
        
        # Apply class weights to the final model training as well
        sample_weight = None
        if use_class_weights:
            sample_weight = compute_sample_weight('balanced', y_data)
        
        self._xgb_model.fit(x_data, y_data, sample_weight=sample_weight)
        self._xgb_model.save_model(out_path / "xgboost.json")

    def xgboost_prediction(self, datapoint: dict) -> dict:
        input_features = self._process_datapoint(datapoint)
        prediction = self._xgb_model.predict(input_features)
        prediction_proba = self._xgb_model.predict_proba(input_features)[0][prediction]

        output = dict()
        output["prediction"] = self._label_encoder.inverse_transform(prediction)[0]
        output["prediction_probability"] = float(prediction_proba[0])

        return output

    def _write_formatted_report(self, report, cm, class_names, out_path):
        """Write a formatted classification report with table and confusion matrix"""
        
        # Save the original JSON report
        with open(out_path / "classification_report.json", "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        
        # Create formatted text report
        with open(out_path / "classification_report_formatted.txt", "w", encoding="utf-8") as file:
            file.write("CLASSIFICATION REPORT\n")
            file.write("=" * 50 + "\n\n")
            
            # Class-wise metrics table
            file.write("CLASS-WISE METRICS\n")
            file.write("-" * 50 + "\n")
            file.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
            file.write("-" * 65 + "\n")
            
            # Sort classes by name for consistent output
            sorted_classes = sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']])
            
            for class_name in sorted_classes:
                metrics = report[class_name]
                file.write(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                          f"{metrics['f1-score']:<10.4f} {metrics['support']:<10.0f}\n")
            
            file.write("-" * 65 + "\n\n")
            
            # Global metrics
            file.write("GLOBAL METRICS\n")
            file.write("-" * 30 + "\n")
            file.write(f"Accuracy: {report['accuracy']:.4f}\n\n")
            
            file.write("Macro Average:\n")
            macro = report['macro avg']
            file.write(f"  Precision: {macro['precision']:.4f}\n")
            file.write(f"  Recall: {macro['recall']:.4f}\n")
            file.write(f"  F1-Score: {macro['f1-score']:.4f}\n\n")
            
            file.write("Weighted Average:\n")
            weighted = report['weighted avg']
            file.write(f"  Precision: {weighted['precision']:.4f}\n")
            file.write(f"  Recall: {weighted['recall']:.4f}\n")
            file.write(f"  F1-Score: {weighted['f1-score']:.4f}\n\n")
            
            # Confusion Matrix
            file.write("CONFUSION MATRIX\n")
            file.write("=" * 50 + "\n")
            file.write("Rows: True labels, Columns: Predicted labels\n\n")
            
            # Calculate optimal column widths
            max_class_name_len = max(len(name) for name in class_names)
            max_num_len = len(str(cm.max()))
            col_width = max(max_class_name_len, max_num_len, 8) + 2  # Add padding
            row_label_width = max(max_class_name_len, 12) + 2  # Add padding
            
            # Header with class names
            file.write(" " * row_label_width)  # Space for row labels
            for name in class_names:
                file.write(f"{name:>{col_width}}")
            file.write("\n")
            
            # Separator line
            total_width = row_label_width + col_width * len(class_names)
            file.write("-" * total_width + "\n")
            
            # Matrix rows
            for i, true_class in enumerate(class_names):
                file.write(f"{true_class:<{row_label_width}}")
                for j in range(len(class_names)):
                    file.write(f"{cm[i][j]:>{col_width}}")
                file.write("\n")
            
            file.write(f"\nTotal samples: {cm.sum()}\n")


    def _create_confusion_matrix_plots(self, cm, class_names, out_path, title_suffix="", filename_prefix="confusion_matrix"):
        """Create confusion matrix plots using seaborn"""
        # Create figure with subplots for both raw and normalized matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw confusion matrix
        sns.heatmap(cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    cbar_kws={'label': 'Count'},
                    ax=ax1)
        
        raw_title = f'Confusion Matrix{title_suffix} (Raw Counts)' if title_suffix else 'Confusion Matrix (Raw Counts)'
        ax1.set_title(raw_title, fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontsize=11)
        ax1.set_ylabel('True Label', fontsize=11)
        ax1.tick_params(axis='x', rotation=45)
        
        # Normalized confusion matrix (by row - true class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        
        # Create annotations that show both percentage and count
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percentage = cm_normalized[i, j] * 100
                row.append(f'{count}\n({percentage:.1f}%)')
            annotations.append(row)
        
        sns.heatmap(cm_normalized, 
                    annot=annotations, 
                    fmt='', 
                    cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    cbar_kws={'label': 'Proportion'},
                    ax=ax2,
                    vmin=0, vmax=1)
        
        norm_title = f'Confusion Matrix{title_suffix} (Normalized)' if title_suffix else 'Confusion Matrix (Normalized by True Class)'
        ax2.set_title(norm_title, fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Label', fontsize=11)
        ax2.set_ylabel('True Label', fontsize=11)
        ax2.tick_params(axis='x', rotation=45)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(out_path / f"{filename_prefix}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save individual normalized matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, 
                    annot=annotations, 
                    fmt='', 
                    cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    cbar_kws={'label': 'Proportion'},
                    vmin=0, vmax=1)
        
        norm_standalone_title = f'Confusion Matrix{title_suffix} (Normalized)' if title_suffix else 'Confusion Matrix (Normalized by True Class)'
        plt.title(norm_standalone_title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(out_path / f"{filename_prefix}_normalized.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrices saved:")
        print(f"  - Combined: {out_path / f'{filename_prefix}.png'}")
        print(f"  - Normalized: {out_path / f'{filename_prefix}_normalized.png'}")



    def _save_subgroup_confusion_matrices(self, y_test, predictions, lengte_test, class_names, out_path):
        """Generate and save separate confusion matrices for each subgroup based on V2 - Lengte"""
        
        # Get unique values of "V2 - Lengte"
        unique_lengte_values = np.unique(lengte_test)
        print(f"Found subgroups in V2 - Lengte: {unique_lengte_values}")
        
        # Create a subdirectory for subgroup matrices
        subgroup_dir = out_path / "subgroup_confusion_matrices"
        subgroup_dir.mkdir(exist_ok=True)
        
        # Generate confusion matrix for each subgroup
        for lengte_value in unique_lengte_values:
            # Filter data for this subgroup
            subgroup_mask = lengte_test == lengte_value
            y_test_subgroup = y_test[subgroup_mask]
            predictions_subgroup = predictions[subgroup_mask]
            
            # Skip if subgroup is too small
            if len(y_test_subgroup) == 0:
                print(f"Warning: No samples found for subgroup '{lengte_value}'")
                continue
                
            print(f"Generating confusion matrix for subgroup '{lengte_value}' with {len(y_test_subgroup)} samples")
            
            # Generate confusion matrix for this subgroup
            unique_labels = np.unique(np.concatenate([y_test_subgroup, predictions_subgroup]))
            cm_subgroup = confusion_matrix(y_test_subgroup, predictions_subgroup, labels=unique_labels)
            
            # Get class names for this subgroup (only include classes that appear)
            subgroup_class_names = [self._label_encoder.classes_[i] for i in unique_labels]
            
            # Save subgroup confusion matrix using unified function
            safe_lengte_name = str(lengte_value).replace("/", "_").replace(" ", "_").replace("-", "_")
            title_suffix = f" - {lengte_value}"
            filename_prefix = f"confusion_matrix_{safe_lengte_name}"
            
            self._create_confusion_matrix_plots(
                cm_subgroup, 
                subgroup_class_names, 
                subgroup_dir, 
                title_suffix,
                filename_prefix
            )
            
            # Generate classification report for this subgroup
            self._save_subgroup_classification_report(
                y_test_subgroup, predictions_subgroup, unique_labels, 
                safe_lengte_name, subgroup_dir, lengte_value
            )

    def _save_subgroup_classification_report(self, y_test_subgroup, predictions_subgroup, unique_labels, safe_lengte_name, subgroup_dir, lengte_value):
        """Save classification report for a subgroup"""
        try:
            subgroup_report = classification_report(
                y_test_subgroup,
                predictions_subgroup,
                target_names=[self._label_encoder.classes_[i] for i in unique_labels],
                labels=unique_labels,
                output_dict=True,
                zero_division=0,
            )
            
            # Save subgroup report
            with open(subgroup_dir / f"classification_report_{safe_lengte_name}.json", "w", encoding="utf-8") as file:
                json.dump(subgroup_report, file, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not generate classification report for subgroup '{lengte_value}': {e}")
