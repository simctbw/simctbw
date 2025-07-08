import pandas as pd
from typing import List, Optional
from pathlib import Path
import datetime as dt
import numpy as np
import sys
import math
import gc
import os

sys.setrecursionlimit(1000000000)


def read_parts(divisor, run_folder_path, name_of_run):
    print("start reading parts")
    
    list_df_dates = [
            pd.read_parquet(path)
            for path in Path.iterdir(run_folder_path)
            if "df_dates" in str(path)
        ]
    
    if len(list_df_dates) > 0:
        df_dates = pd.concat(list_df_dates)

        df_dates.to_parquet(
            Path.joinpath(run_folder_path, f"df_dates_{str(name_of_run)}.parquet"), index=False
        )

        del df_dates
        gc.collect()

    for part in range(divisor):
        print(part / divisor)
        dfs = []
        for i, path in enumerate(Path.iterdir(run_folder_path)):
            #if os.path.getsize(path) > 0:
            if True:
                if (
                    i % divisor == part
                    and "df_agents" in str(path)
                    and "_PART_" not in str(path)
                ):
                    df = pd.read_parquet(path)
                    model_index = int(str(path).split("_agents_m")[1].split("-")[0])
                    df["model_index"] = model_index
                    dfs.append(df)


        if len(dfs) > 0:
            df_agents_part = pd.concat(dfs)
            df_agents_part.to_parquet(
                Path.joinpath(
                    run_folder_path,
                    f"df_agents_{str(name_of_run)}_PART_{str(part)}.parquet",
                ),
                index=False,
            )
            del df_agents_part
        del dfs
        del df
        gc.collect()

    gc.collect()

    for path in Path.iterdir(run_folder_path):
        if "df_agents" in str(path):
            if "_PART_" not in str(path):
                os.remove(path)

    for path in Path.iterdir(run_folder_path):
        if "df_dates" in str(path):
            if f"df_dates_{str(name_of_run)}.parquet" not in str(path):
                os.remove(path)

    for path in Path.iterdir(run_folder_path):
        if "df_params" in str(path):
            os.remove(path)

    # df_agents.to_parquet(Path.joinpath(run_folder_path, f"df_agents_{str(NAME_OF_RUN)}.parquet"), index=False)


def predict_caused_infections(age_rki, household_size):
    """Calculates a value representing the number of caused infections as predicted by the regression model."""
    # effect of age
    b_age = {
        0: 0,
        5:  0.4838, 
        15: 1.2509,
        35: 1.3219, 
        60: 0.6673, 
        80: 0.1121,
    }[age_rki]

    # effect of household size
    b_hhs = 0.2430

    # regression formula
    pred_value = b_age + household_size * b_hhs

    return pred_value


def group_age_like_rki(age):
    if age <= 4:
        return 0
    elif age <= 14:
        return 5
    elif age <= 34:
        return 15
    elif age <= 59:
        return 35
    elif age <= 79:
        return 60
    else:
        return 80


def recode_age_group_rki_to_inc_rank_sim(age_group_rki):
    if age_group_rki == 0:
        return 6
    elif age_group_rki == 5:
        return 4
    elif age_group_rki == 15:
        return 2
    elif age_group_rki == 35:
        return 1
    elif age_group_rki == 60:
        return 3
    elif age_group_rki == 80:
        return 5


def recode_age_group_rki_to_inc_rank_emp(age_group_rki):
    if age_group_rki == 0:
        return 6
    elif age_group_rki == 5:
        return 3
    elif age_group_rki == 15:
        return 1
    elif age_group_rki == 35:
        return 2
    elif age_group_rki == 60:
        return 4
    elif age_group_rki == 80:
        return 5


def recode_age_group_rki_to_age_rank(age_group_rki):
    if age_group_rki == 0:
        return 5
    elif age_group_rki == 5:
        return 3
    elif age_group_rki == 15:
        return 2
    elif age_group_rki == 35:
        return 1
    elif age_group_rki == 60:
        return 4
    elif age_group_rki == 80:
        return 6


def get_rmse(col1, col2, divide_e_by_col1=False):
    e = col1 - col2
    if divide_e_by_col1:
        e = e / col1
    se = e * e
    mse = se.mean()
    rmse = math.sqrt(mse)
    return rmse


def print_header(text):
    print("")
    print("")
    print("______________________________________")
    print(text)
    print("______________________________________")
    print("")


def drop_incomplete_households(
    df: pd.DataFrame, columns: List[str], household_level=True
) -> pd.DataFrame:
    """
    Drops all rows (people) that belong to a household
    that contains at least one person with a missing value on one of the given columns.
    """
    df = df.copy()

    n_rows_pre = len(df)

    for col in columns:
        # check people for missing values
        df["data_is_complete"] = 1
        df.loc[df[col].isna(), "data_is_complete"] = 0

        # check households for people with missing values and keep only people living in "complete" households
        df["hh_data_is_complete"] = df.groupby("hid")["data_is_complete"].transform(min)

        if household_level:
            # keep only people living in "complete" households
            df = df[df["hh_data_is_complete"] == 1]
        else:
            df = df[df["data_is_complete"] == 1]

        df = df.drop(["data_is_complete", "hh_data_is_complete"], axis=1)

        df = df.reset_index(drop=True)

    n_rows_post = len(df)
    n_rows_dropped = n_rows_pre - n_rows_post
    perc_dropped = round((n_rows_dropped / n_rows_pre) * 100, 2)

    print("n rows dropped:", n_rows_dropped)
    print("percentage dropped:", perc_dropped)

    return df


def load_partial_soep_dataset(
    path: Path,
    rename_columns: Optional[dict] = None,
    keep_columns: Optional[list] = None,
    filter_year: Optional[int] = None,
    drop_duplicates: bool = False,
    drop_essential_missings: bool = True,
    keep_renamed_columns=True,
) -> pd.DataFrame:
    print("Loading ...", path)

    df = pd.read_stata(path, convert_categoricals=False)

    if filter_year is not None:
        df = df[df["syear"] == filter_year]

    if drop_essential_missings:
        for col in ["hid", "pid", "syear"]:
            if col in df.columns:
                df = df[~df[col].isna()]

    if rename_columns is not None:
        df = df.rename(columns=rename_columns)

    if keep_columns is not None or keep_renamed_columns:
        if not isinstance(keep_columns, list) and keep_renamed_columns:
            keep_columns = []

        if rename_columns is not None and keep_renamed_columns:
            for col_name in rename_columns.values():
                if col_name not in keep_columns:
                    keep_columns.append(col_name)

        standard_columns = [
            name
            for name in ["hid", "pid", "syear"]
            if name in df.columns and name not in keep_columns
        ]
        keep_columns.extend(standard_columns)

        df = df[keep_columns]

    if drop_duplicates:
        len_df = len(df)
        df = df.drop_duplicates(subset=["syear", "pid"])
        print(f"{len_df - len(df)} duplicates dropped.")
    return df


def str_time() -> str:
    return (
        str(dt.datetime.now())
        .replace(".", "")
        .replace(":", "")
        .replace(" ", "")
        .replace("-", "")
    )


def dates_between(date1: dt.date, date2: dt.date) -> List[dt.date]:
    day_diff = (date2 - date1).days
    dates = [date1 + dt.timedelta(days=i) for i in range(day_diff + 1)]
    return dates


def group_it(
    value, start, step, n_steps, return_value="index", summarize_highest=False
):
    assert value >= start, (
        f"The value {value} is smaller than the smallest lower bound {start}."
    )

    for i in range(n_steps):
        lower_bound = start + step * i
        upper_bound = lower_bound + step

        if lower_bound <= value:
            if return_value == "index":
                new_value = i

            elif return_value == "lower_bound":
                new_value = lower_bound

            elif return_value == "range":
                new_value = (lower_bound, upper_bound)

        if not summarize_highest:
            if i == n_steps + 1:
                if value > upper_bound:
                    new_value = np.nan

    return new_value
