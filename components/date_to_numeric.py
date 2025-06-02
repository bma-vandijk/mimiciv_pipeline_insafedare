import os
import pandas as pd


def date_to_numeric(
    input_path: str,
    output_path: str,
    columns: list,
):
    """
    Cast datetime columns to a numeric format, i.e., (fractional) proleptic Gregorian ordinal.
    This makes downstream feature engineering and modelling much easier, as most data processing cannot handle datetimes.
    """
    df = pd.read_parquet(input_path)

    def date_to_floating_ordinal(dt):
        """
        Calculates fractional days of datetime, as pd.toordinal() rounds to full days.
        """
        return (
            dt.toordinal()
            + (
                dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)
            ).total_seconds()
            / 86400
        )

    for col in columns:
        df[col] = pd.to_datetime(df[col])
        df[col] = df[col].apply(date_to_floating_ordinal)

    df.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    columns = os.environ.get("columns", "").split(",")

    date_to_numeric(
        input_path=input_path,
        output_path=output_path,
        columns=columns
    ) 