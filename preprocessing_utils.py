
import pandas as pd


def apply_binary_model_mapping(df: pd.DataFrame, target_column: str = "V2 - Licentie Nodig") -> pd.DataFrame:
    """
    Map labels for binary model: "Eigen Werk" and "Open Access" -> "Nee"
    
    Args:
        df: DataFrame with the data
        target_column: Name of the column containing the labels
        
    Returns:
        DataFrame with mapped labels
    """
    df = df.copy()
    mapping = {
        "Eigen Werk": "Nee",
        "Open Access": "Nee",
        "Ja": "Ja",
        "Nee": "Nee"
    }
    df[target_column] = df[target_column].map(mapping)
    print(f"Applied binary model mapping. New label distribution:")
    print(df[target_column].value_counts())
    return df


def remove_nee_labels(df: pd.DataFrame, target_column: str = "V2 - Licentie Nodig") -> pd.DataFrame:
    """
    Remove all rows with "Nee" label
    
    Args:
        df: DataFrame with the data
        target_column: Name of the column containing the labels
        
    Returns:
        DataFrame with "Nee" labels removed
    """
    df_filtered = df[df[target_column] != "Nee"].copy()
    print(f"Removed 'Nee' labels. Dropped {len(df) - len(df_filtered)} rows.")
    print(f"New label distribution:")
    print(df_filtered[target_column].value_counts())
    return df_filtered


def map_nee_to_eigenwerk(df: pd.DataFrame, target_column: str = "V2 - Licentie Nodig") -> pd.DataFrame:
    """
    Map "Nee" labels to "Eigen Werk"
    
    Args:
        df: DataFrame with the data
        target_column: Name of the column containing the labels
        
    Returns:
        DataFrame with "Nee" labels mapped to "Eigen Werk"
    """
    df = df.copy()
    mapping = {
        "Eigen Werk": "Eigen Werk",
        "Open Access": "Open Access", 
        "Ja": "Ja",
        "Nee": "Eigen Werk"
    }
    df[target_column] = df[target_column].map(mapping)
    print(f"Mapped 'Nee' to 'Eigen Werk'. New label distribution:")
    print(df[target_column].value_counts())
    return df


def apply_preprocessing_option(df: pd.DataFrame, option: str, target_column: str = "V2 - Licentie Nodig") -> pd.DataFrame:
    """
    Apply preprocessing option to the DataFrame
    
    Args:
        df: DataFrame with the data
        option: Preprocessing option ("binary_model", "remove_nee", "nee_to_eigenwerk", or None)
        target_column: Name of the column containing the labels
        
    Returns:
        DataFrame with preprocessing applied
    """
    if option is None:
        return df
    elif option == "binary_model":
        return apply_binary_model_mapping(df, target_column)
    elif option == "remove_nee":
        return remove_nee_labels(df, target_column)
    elif option == "nee_to_eigenwerk":
        return map_nee_to_eigenwerk(df, target_column)
    else:
        raise ValueError(f"Unknown preprocessing option: {option}. "
                        f"Valid options are: 'binary_model', 'remove_nee', 'nee_to_eigenwerk', or None")
