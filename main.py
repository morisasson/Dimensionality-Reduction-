import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a file path.
    """
    try:
        if filepath.endswith(".csv"):
            return pd.read_csv(filepath)
        elif filepath.endswith((".xls", ".xlsx")):
            return pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        raise ValueError(f"Failed to load file: {e}")


def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
    """
    Group and aggregate data by a specified column.
    """
    numeric_columns = df.select_dtypes(include=[np.number])
    numeric_columns[group_by_column] = df[group_by_column]

    if numeric_columns.shape[1] < 2:
        raise ValueError("Not enough numeric columns available after filtering.")

    df_grouped = numeric_columns.groupby(group_by_column).agg(agg_func)
    return df_grouped
def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Remove sparse columns with a sum below a specified threshold.
    """
    numeric_columns = df.select_dtypes(include=['number'])
    column_sums = numeric_columns.sum()
    numeric_columns_to_keep = column_sums[column_sums > threshold].index

    if len(numeric_columns_to_keep) < 2:
        print("Filtered data is empty or not enough numeric columns available after removing sparse columns.")
        print("Numeric columns to keep:", numeric_columns_to_keep)
        print("Threshold:", threshold)
        raise ValueError(
            "Filtered data is empty or not enough numeric columns available after removing sparse columns.")

    return df[list(numeric_columns_to_keep) + list(df.select_dtypes(exclude=['number']).columns)]


def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    """
    Perform PCA using SVD for large datasets.
    """
    # Extract numeric data
    numeric_data = df.drop(columns=meta_columns).select_dtypes(include=[np.number])
    print("Shape of numeric data before PCA:", numeric_data.shape)

    # Ensure there are numeric columns to process
    if numeric_data.shape[1] < 2:
        print("Numeric data available for PCA:", numeric_data.columns)
        raise ValueError("Not enough numeric columns available for PCA.")

    # Standardize the data (zero mean, unit variance)
    standardized_data = (numeric_data - numeric_data.mean()) / numeric_data.std()

    # Perform SVD
    U, S, Vt = np.linalg.svd(standardized_data, full_matrices=False)

    # Select the top components
    #reduced_data = np.dot(U[:, :num_components], np.diag(S[:num_components]))
    reduced_data = np.dot(standardized_data, Vt.T[:, :num_components])

    # Create a DataFrame with the reduced data and meta columns
    reduced_df = pd.DataFrame(reduced_data, columns=[f"PC{i + 1}" for i in range(num_components)])

    # Flip the values of PC2
    reduced_df['PC2'] = reduced_df['PC2'] * -1

    return reduced_df.join(df[meta_columns].reset_index(drop=True))

