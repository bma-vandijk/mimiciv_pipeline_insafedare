import os
import pandas as pd


def ingest_data(
    input_path: str,
    output_path: str,
    usecols: list = [],
):
    """
    Reads data and writes to shared volume mount in parquet.
    Currently only support csv and gzip compression
    """
    if len(usecols) == 0:
        usecols = None
    split = input_path.split(".")
    if "gz" in split:
        compression = "gzip"
    else:
        compression = None
    if "csv" in split:
        df = pd.read_csv(input_path, compression=compression, usecols=usecols, low_memory=False)
    elif "txt" in split:
        df = pd.read_csv(
            input_path, compression=compression, usecols=usecols, delimiter="\t"
        )

    # Add informative df name to output path
    df_name = input_path.split("/")[-1]
    df_name = df_name.split('.')[0]

    df.to_parquet(os.path.join(output_path, f"{df_name}.parquet"))

if __name__ == "__main__":
    usecols = os.environ.get("usecols", "")
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    
    usecols = usecols.split(",") if usecols else [] # Convert comma-separated string to list, empty string becomes empty list

    ingest_data(usecols=usecols, input_path=input_path, output_path=output_path)