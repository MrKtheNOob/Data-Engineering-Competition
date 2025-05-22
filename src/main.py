import pandas as pd
from datetime import datetime
import numpy as np


def load_user_data() -> pd.DataFrame:
    df = pd.read_json("../users.json", dtype={"national_id": str})
    return df

def load_transaction_data() -> pd.DataFrame:
    df = pd.read_json("../transactions.json")
    return df

def clean_dates(value, output_format: str = "%d/%m/%Y %H:%M:%S"):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    formats = ["%d-%m-%Y-%H-%M", "%Y/%m/%d %H:%M"]  # formats that are in the file
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)  # tries to parse the format with fmts
            return parsed.strftime(output_format) # returns with the desired format if the parsing works
        except ValueError:
            continue

    return np.nan # returns NaN if the value is null or does not have one of the formats in fmt

def correct_to_uppercase(string: str):
    return string.upper() if not string.isupper() else string

def hide_email(email: str) -> str:
    at_index = email.find("@")
    hidden_part = "*" * (at_index - 1)
    return email[0] + hidden_part + email[at_index:]

def hide_national_id(id: str):
    id = str(id)
    return id[:3] + "X" * len(id[3:])

def flatten_location(users: pd.DataFrame) -> pd.DataFrame:
    location_data = pd.json_normalize(users["location"].to_list())
    users = users.drop(columns=["location"])  # drop the location column
    users = pd.concat(
        [users, location_data], axis=1
    )  # replace location by its nested attributes as columns
    return users


users: pd.DataFrame = load_user_data()


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    users = flatten_location(users)
    users = users.drop_duplicates()
    users["account_created"] = users["account_created"].apply(clean_dates)
    users["user_id"] = users["user_id"].apply(correct_to_uppercase)
    users["email"] = users["email"].apply(hide_email)
    users["national_id"] = users["national_id"].apply(hide_national_id)

    return users



clean_users(users).to_json("../cleaned_files/cleaned_users.json",orient="records", indent=2, force_ascii=False)
clean_users(users).to_csv("../cleaned_files/cleaned_users.json",index=False)