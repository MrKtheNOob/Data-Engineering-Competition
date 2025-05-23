import pandas as pd
from datetime import datetime
import numpy as np
from pathlib import Path

DIRTY_USERS_FILE = "../dirty_files/users.json"
DIRTY_TRANSACTIONS_FILE = "../dirty_files/transactions.json"
OUTPUT_DIRECTORY = "../cleaned_files/"

# Helper functions
def load_user_data(path:str) -> pd.DataFrame:
    df = pd.read_json(path, dtype={"national_id": str})
    return df

def load_transaction_data(path:str) -> pd.DataFrame:
    df = pd.read_json(path)
    return df

def clean_dates(value, output_format: str = "%d/%m/%Y %H:%M:%S") -> str | float:
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    formats = ["%d-%m-%Y-%H-%M", "%Y/%m/%d %H:%M"]  # formats that are in the file
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)  # tries to parse the format with fmts
            return parsed.strftime(output_format)  # returns with the desired format if the parsing works
        except ValueError:
            continue

    return np.nan  # returns NaN if the value is null or does not have one of the formats in fmt

def correct_to_uppercase(string: str):
    return string.upper() if not string.isupper() else string

def hide_email(email: str) -> str:
    at_index = email.find("@")
    hidden_part = "*" * (at_index - 1)
    return email[0] + hidden_part + email[at_index:]

def hide_national_id(id: str) -> str:
    id = str(id)
    return id[:3] + "X" * len(id[3:])

def flatten_location(users: pd.DataFrame) -> pd.DataFrame:
    location_data = pd.json_normalize(users["location"].to_list())
    users = users.drop(columns=["location"])  # drop the location column
    users = pd.concat([users, location_data], axis=1)  # replace location by its nested attributes as columns
    return users

def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    users = flatten_location(users)
    users = users.drop_duplicates()
    users["account_created"] = users["account_created"].apply(clean_dates)
    users["user_id"] = users["user_id"].apply(correct_to_uppercase)
    users["email"] = users["email"].apply(hide_email)
    users["national_id"] = users["national_id"].apply(hide_national_id)

    return users

def clean_transactions(transactions: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    pd.set_option('display.float_format', '{:.2f}'.format)  # set numbers to limit 2 decimal digits
    transactions.rename(columns={"tx_id": "ID"}, inplace=True)  # rename tx_id to ID
    transactions["timestamp"] = transactions["timestamp"].apply(clean_dates)
    transactions["user_id"] = transactions["user_id"].apply(correct_to_uppercase)
    users_id_set = set(users["user_id"])
    transactions = transactions[transactions["user_id"].isin(users_id_set)]
    
    return transactions

def clean_data(users_path: str, transactions_path: str, output_path: str) -> pd.DataFrame:
    Path(output_path).mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    # Loading files
    users: pd.DataFrame = load_user_data(users_path)
    transactions: pd.DataFrame = load_transaction_data(transactions_path)
    # Cleaning
    users=clean_users(users)
    transactions=clean_transactions(transactions, users)
    # Merge and save to csv
    result:pd.DataFrame= users.merge(transactions, on="user_id", how="inner")
    result.to_csv(Path(output_path) / "output.csv", index=False)    
    
    return result

if __name__ == "__main__":
    clean_data(DIRTY_USERS_FILE,DIRTY_TRANSACTIONS_FILE,OUTPUT_DIRECTORY)