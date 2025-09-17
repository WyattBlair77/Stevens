from __future__ import annotations

import locale
from typing import Any, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import os
from scipy.stats import chi2_contingency
from sklearn import metrics, model_selection
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_classif,
)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)


ARBITRARY_NUM_ESTIMATORS_FOR_FEATURE_SELECTION = 100
ARBITRARY_NUM_FEATURES_TO_SELECT = 10
DEFAULT_CV_FOLDS = 5


import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self: DataFrameImputer):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        Note: this class is not my own. Originally adapted from "sveitser", found on this link:
        https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
        """
    def fit(self, X: pd.DataFrame | pd.Series, y: pd.Series | np.ndarray = None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X: pd.DataFrame | pd.Series, y: pd.Series | np.ndarray = None):
        return X.fillna(self.fill)


class StringStandardScaler(StandardScaler):
    """
    Applies standard scaling. Before any computation, it
    first converts the data (assumed to be 1D strings) to
    integers.
    """

    def _parse_comma_separated_int(
            self,
            decimal_val: Union[str, np.ndarray],
            locale_category: str = 'de_DE'  # because the BMI is encoded in the European style (i.e., commas function as decimals)
        ) -> int:
        """Converts a Python integer encoded as a string into an int."""
        if isinstance(decimal_val, np.ndarray):
            decimal_val = decimal_val[0]
        _ = locale.setlocale(locale.LC_ALL, locale_category)
        return locale.atof(decimal_val)

    def fit(
        self: StringStandardScaler,
        X: pd.Series,
        y=None,
        sample_weight=None
    ) -> StringStandardScaler:
        """
        Custom fit() operation. Ignore the `y` parameter.
        """
        X = X.to_numpy().squeeze()
        X_transformed = np.array(
            list(map(self._parse_comma_separated_int, X))
        ).reshape(-1, 1)
        return super().fit(X_transformed, y, sample_weight)

    def transform(
            self: StringStandardScaler,
            X: pd.Series,
        ) -> np.ndarray:
        """
        Custom transform() operation.
        """
        X = X.to_numpy().squeeze()
        X_transformed = np.array(
            list(map(self._parse_comma_separated_int, X))
        ).reshape(-1, 1)
        return super().transform(X_transformed, copy=True)


def cramer_v(two_feature_cols: np.ndarray) -> float:
    """
    Used to compute multicolinearity between categorical features.

    Cramer's V implementation in Python, adapted from ChatGPT:
    https://chat.openai.com/share/269f1ad3-93b2-4942-8797-932356c616f2.
    """
    confusion_matrix = pd.crosstab(two_feature_cols[:, 0], two_feature_cols[:, 1])
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def extract_best_model_from_grid_search_results(results: dict) -> BaseEstimator:
    """
    Retrieves the best performing model outputted by a cross-validated training run.

    Parameters:
        results(dict): this is the output to calling `GridSearchCV.fit().`

    Returns: Estimator - a fitted classifier model of some type.
    """
    _, best_clf_model = results.best_estimator_.steps[-1]
    return best_clf_model


def run_classification_experiment(
    data: tuple[Union[np.ndarray, pd.DataFrame, pd.Series]],
    clf: BaseEstimator,
    preprocessor: ColumnTransformer,
    param_grid: dict[str, list[Union[float, int, bool, str]]] = {},
) -> dict:
    """
    Attempt to optimize a classification model on a dataset.

    Performs data preprocessing, hyperparameter tuning, and
    cross-validation to arrive at a well-performing and
    generalized ML model. Then performs analysis in the form of
    printing the accuracy, error rate, precision, recall,
    f1-score, confusion matrix, and ROC AUC.

    Example Usage:
        >>> X, y = ...
        >>> clf = ...
        >>> preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), ("Location", "stress"))])
        >>> param_grid = {"classifier__criterion": ["gini", "entropy", "log_loss"]}
        >>> results = util.run_classification_experiment(
            ... train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True),
            ... clf,
            ... preprocessor,
            ... param_grid,
        ... )

    Parameters:
        data(tuple(X_train, X_test, y_train, y_test):
            assumes train_test_split has already been called, but not preprocessing
        clf(Estimator): some instantiated object that can do classification
        preprocessor(ColumnTransformer):
            a collection that specifies what preprocessing steps you want to do, and
            which columns in the dataset where you specifically want them to be applied
        param_grid(dict): the search space for the grid search. Will depend on `type(clf)`
            Be sure to prefix all the keys in this dict with "classifier__"

    Returns: dict: the results outputted by `GridSearchCV`. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
    """
    X_train, X_test, y_train, y_test = data

    # run cross validation, nested within grid search
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    # make sure all hyperparameters named correctly
    for hparam, value in param_grid.items():
        if isinstance(hparam, str) and hparam.startswith("classifier__") is False:
            param_grid[f"classifier__{hparam}"] = param_grid.pop(hparam)

    grid_searcher = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        refit="f1",
    )

    # analyze and visualize the results
    results = grid_searcher.fit(X_train, y_train)
    best_clf_model = extract_best_model_from_grid_search_results(results)
    print(f"The best model set this config: {best_clf_model.__dict__}")

    y_pred = results.best_estimator_.predict(X_test)

    print(f"====== Accuracy and Error Rate ==============")
    acc_score = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {acc_score * 100}%.")
    print(f"Error rate: {(1 - acc_score) * 100}%.")

    print(f"========== Classification Report ============")
    print(metrics.classification_report(y_test, y_pred))

    print(f"============ Confusion Matrix ==============")
    _ = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.show()

    print(f"=============== ROC and AUC ================")
    _ = RocCurveDisplay.from_estimator(results.best_estimator_, X_test, y_test)

    y_score = None
    if hasattr(results.best_estimator_, "predict_proba"):
        # let's go off of probability estimates
        y_score = results.best_estimator_.predict_proba(X_test)[:, 1]
    elif hasattr(results.best_estimator_, "decision_function"):
        # let's go off of decision values
        y_score = results.best_estimator_.decision_function(X_test)

    roc_auc = metrics.roc_auc_score(y_test, y_score)
    gini_coefficient = 2 * roc_auc - 1
    # this is normalization s.t. random guessing will be have 0 in expectation, and it is upper bounded by 1.
    print(f"Gini coefficient (normalized ROC AUC): {gini_coefficient}.")
    plt.show()

    return results


def select_features(
    data: tuple[Union[np.ndarray, pd.DataFrame, pd.Series]],
    preprocessor: ColumnTransformer,
    method: Union[
        Literal["rf_feature_importances"],
        Literal["refine_with_ranking"]
    ] = "rf_feature_importances",
    num_features_to_select: int = ARBITRARY_NUM_FEATURES_TO_SELECT,
    num_trees: int = ARBITRARY_NUM_ESTIMATORS_FOR_FEATURE_SELECTION,
    index_of_categorical_preprocessor: int = -1,
    cross_val_folds: int = DEFAULT_CV_FOLDS,
) -> list[str]:
    """TODO[Zain]: add docstring"""

    ### HELPER(S)
    def _get_list_of_transformed_feature_names(
            initial_feature_list: list[str],
        ) -> np.ndarray:
        ...
        one_hot_encoder: OneHotEncoder = preprocessor.transformers_[index_of_categorical_preprocessor][1]
        categorical_non_binary_cols = one_hot_encoder.feature_names_in_
        one_hot_transformed_names = one_hot_encoder.get_feature_names_out(categorical_non_binary_cols).tolist()
        index_of_first_categorical_feature = initial_feature_list.index(categorical_non_binary_cols[0])
        index_of_last_categorical_feature = index_of_first_categorical_feature + len(categorical_non_binary_cols) - 1
        feature_names = (
            initial_feature_list[:index_of_first_categorical_feature] +
            one_hot_transformed_names +
            initial_feature_list[index_of_last_categorical_feature+1:]
        )
        return np.array(feature_names)

    def _evaluate_selected_features_with_cross_val(
        X_train_transformed: np.ndarray,
        y_train: np.ndarray,
        selected_feature_indices: np.ndarray
    ) -> None:
        results: dict[str, np.ndarray] = model_selection.cross_validate(
            RandomForestClassifier(num_trees),
            X_train_transformed[:, selected_feature_indices],
            y_train,
            cv=cross_val_folds,
            scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'),
            return_train_score=True,
        )
        print(f"======= Selected {k} Features To Train a Random Forest ({num_trees} Estimators, '{method}' method) =======")
        print(f"Accuracy: {results['test_accuracy'].mean()}")
        print(f"Precision: {results['test_precision'].mean()}")
        print(f"Recall: {results['test_recall'].mean()}")
        print(f"F1-Score: {results['test_f1'].mean()}")
        print(f"ROC AUC: {results['test_roc_auc'].mean()}")
        print(f"Top {k} Feature Names: {selected_feature_names}")

    ### DRIVER
    X_train, _, y_train, _ = data
    k = num_features_to_select
    selected_feature_names = []

    # Compute Feature Importance
    if method == "rf_feature_importances":

        # get a list containing all the expanded column names
        feature_names = list(X_train.columns)
        _ = preprocessor.fit(X_train)

        if index_of_categorical_preprocessor > -1:
            feature_names = _get_list_of_transformed_feature_names(feature_names)

        # train the RF
        rf = RandomForestClassifier(n_estimators=num_trees, random_state=42)
        X_train_transformed = preprocessor.transform(X_train)
        rf.fit(X_train_transformed, y_train)

        # cross validate to see how well these features work
        feature_importances = rf.feature_importances_
        selected_feature_indices = np.argsort(feature_importances)[-k:]
        selected_feature_names = feature_names[selected_feature_indices].tolist()

        _evaluate_selected_features_with_cross_val(
            X_train_transformed, y_train, selected_feature_indices
        )

        # make final output dict
        selected_feature_names_with_values = dict(zip(
            selected_feature_names,
            feature_importances[selected_feature_indices]
        ))

    # Recursively eliminate not so useful features
    elif method == "refine_with_ranking":
        # get a list containing all the expanded column names
        feature_names = list(X_train.columns)
        _ = preprocessor.fit(X_train)

        if index_of_categorical_preprocessor > -1:
            feature_names = _get_list_of_transformed_feature_names(feature_names)

        # train the RF
        rf = RandomForestClassifier(n_estimators=num_trees, random_state=42)
        selector = RFE(rf, n_features_to_select=k, step=1)
        X_train_transformed = preprocessor.transform(X_train)
        _ = selector.fit_transform(X_train_transformed, y_train)
        selected_feature_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]
        selected_feature_ranks = selector.ranking_[selected_feature_indices]  # admittedly this isn't going to be useful, but helps to keep a consistent output

        # cross validate to see how well these features work
        _evaluate_selected_features_with_cross_val(
            X_train_transformed, y_train, selected_feature_indices
        )

        # make final output dict
        selected_feature_names_with_values = dict(zip(
            selected_feature_names,
            selected_feature_ranks
        ))

    else:
        raise ValueError(f"Invalid method set to = '{method}'. Available options are: []")


    # return the list of features to select
    return selected_feature_names_with_values


def min_max_normalization(series):

    minimum, maximum = series.min(), series.max()
    normalized_data = (series - minimum) / (maximum - minimum)

    return normalized_data


def z_score_standardization(series):

    mean, std = series.mean(), series.std()
    normalized_data = (series - mean) / std

    return normalized_data


def load_miscarriage_data():

    pass


def load_heart_disease_data():
    
    pass


def load_airline_data(load_from_disk=True, train_test_split=0.7):

    '''
    This function will server a universal way for us to all load data in an identical manner.

    parameters:

        load_from_disk (bool):        whether or not to load the already preprocessed data in the
                                      preprocessed_<train/test>_data.csv files. if False, load raw
                                      data, perform preprocessing, and save for next time

        train_test_split (float):     percent of data to allocate to train and percent to allocate to 
                                      test. if load_from_disk is True, this will have no affect

    dataset has following features:
    ==================================================================================================================
    DEP_DEL15: 			                  ~TARGET~ Binary of a departure delay over 15 minutes (1 is yes)
    MONTH:				                  Month
    DAY_OF_WEEK:	                      Day of Week
    DISTANCE_GROUP:	                      Distance group to be flown by departing aircraft
    DEP_BLOCK:			                  Departure block
    SEGMENT_NUMBER:		                  The segment that this tail number is on for the day
    CONCURRENT_FLIGHTS:		              Concurrent flights leaving from the airport in the same departure block
    NUMBER_OF_SEATS:		              Number of seats on the aircraft
    CARRIER_NAME:			              Carrier
    AIRPORT_FLIGHTS_MONTH:		          Avg Airport Flights per Month
    AIRLINE_FLIGHTS_MONTH:		          Avg Airline Flights per Month
    AIRLINE_AIRPORT_FLIGHTS_MONTH:	      Avg Flights per month for Airline AND Airport
    AVG_MONTHLY_PASS_AIRPORT:	          Avg Passengers for the departing airport for the month
    AVG_MONTHLY_PASS_AIRLINE:	          Avg Passengers for airline for month
    FLT_ATTENDANTS_PER_PASS:	          Flight attendants per passenger for airline
    GROUND_SERV_PER_PASS:		          Ground service employees (service desk) per passenger for airline
    PLANE_AGE:			                  Age of departing aircraft
    DEPARTING_AIRPORT:		              Departing Airport
    LATITUDE:			                  Latitude of departing airport
    LONGITUDE:			                  Longitude of departing airport
    PREVIOUS_AIRPORT:		              Previous airport that aircraft departed from
    PRCP:				                  Inches of precipitation for day
    SNOW:				                  Inches of snowfall for day
    SNWD:				                  Inches of snow on ground for day
    TMAX:				                  Max temperature for day
    AWND:				                  Max wind speed for day
    ==================================================================================================================
    '''

    raw_fp = './data/airline/full_data_flightdelay.csv'
    preproccessed_train_fp = './data/airline/preprocessed_train_data.csv'
    preproccessed_test_fp = './data/airline/preprocessed_test_data.csv'

    # check if preprocessed data has been saved locally already
    if all([
        os.path.exists(preproccessed_train_fp),
        os.path.exists(preproccessed_test_fp),
    ]) and load_from_disk:

        train = pd.read_csv(preproccessed_train_fp)
        test = pd.read_csv(preproccessed_test_fp)

        return train, test

    else:
        # Load full DataFrame
        data = pd.read_csv(raw_fp)

        # Target / Feature Names
        target = 'DEP_DEL15'
        features = data.columns.to_list(); features.remove(target)

        # Take string-cols and target-col and make them numerical
        str_cols = list(data.dtypes[data.dtypes == object].index) + [target]
        for str_col in tqdm(str_cols, total=len(str_cols), desc='Enumerating Categorical Data'):

            vals = data[str_col].unique()
            data.replace(
                {
                    col_val: i
                    for i, col_val in enumerate(vals)
                },
                inplace=True
            )

        # normalize using min-max normalization
        data[features] = min_max_normalization(data[features])
        train = data.sample(frac=train_test_split)
        test = data.drop(index=train.index)

        # save so it's faster next time
        train.to_csv(preproccessed_train_fp)
        test.to_csv(preproccessed_test_fp)

        return train, test


def test_classifier(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
    
    results = {}

    predictions = model.predict(X_test)
    correct = y_test == predictions
    correct_counts = correct.value_counts()[True]

    acc = correct_counts / len(y_test.index)

    results.update({
        'Accuracy': acc,
        'F1-Score': f1_score(y_true=y_test, y_pred=predictions)
    })

    return results

  
def transform_heart_disease(
    path: str,
    mode: Union[Literal["zain"], str]
) -> Any:
    ### HELPER(S)
    def _zain_heart_disease_transforms() -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        ColumnTransformer,
    ]:
        categorical_unordered_cols = [
            "State",
            "Sex",
            "RaceEthnicityCategory",
        ]

        binary_or_categorical_ordered_cols = [
            "GeneralHealth",
            "LastCheckupTime",
            "PhysicalActivities",
            "RemovedTeeth",
            "HadAngina",
            "HadStroke",
            "HadAsthma",
            "HadSkinCancer",
            "HadCOPD",
            "HadDepressiveDisorder",
            "HadKidneyDisease",
            "HadArthritis",
            "HadDiabetes",
            "DeafOrHardOfHearing",
            "BlindOrVisionDifficulty",
            "DifficultyConcentrating",
            "DifficultyWalking",
            "DifficultyDressingBathing",
            "DifficultyErrands",
            "SmokerStatus",
            "ECigaretteUsage",
            "ChestScan",
            "AgeCategory",
            "AlcoholDrinkers",
            "HIVTesting",
            "FluVaxLast12",
            "PneumoVaxEver",
            "TetanusLast10Tdap",
            "HighRiskLastYear",
            "CovidPos",
        ]

        continuous_cols = [
            "SleepHours",
            "MentalHealthDays",
            "PhysicalHealthDays",
            "HeightInMeters",
            "WeightInKilograms",
            "BMI",
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("ordered", OrdinalEncoder(), binary_or_categorical_ordered_cols),
                ("unordered", OneHotEncoder(), categorical_unordered_cols),
                ("cont", StandardScaler(), continuous_cols),
            ]
        )

        df = pd.read_csv(path)

        TARGET_COL = "HadHeartAttack"

        df_no_nan_targets = df.dropna(subset=[TARGET_COL])
        df_no_nan_targets = df_no_nan_targets.drop_duplicates(keep="first")

        features = list(df_no_nan_targets.columns)
        features.remove(TARGET_COL)

        X = df_no_nan_targets[features]
        y = df_no_nan_targets[TARGET_COL]

        X_no_nan = DataFrameImputer().fit_transform(X)
        y_no_nan_encoded = LabelEncoder.fit_transform(y)

        return X_no_nan, y_no_nan_encoded, preprocessor

    ### DRIVER:

    if mode == "zain":
        return _zain_heart_disease_transforms()

