import pandas as pd
from datetime import datetime
import numpy as np
from pathlib import Path

DIRTY_USERS_FILE = "../dirty_files/users.json"
DIRTY_TRANSACTIONS_FILE = "../dirty_files/transactions.json"
OUTPUT_DIRECTORY = "../cleaned_files/"

def clean_data(users_path: str, transactions_path: str, output_path: str) -> pd.DataFrame:
    """
    Nettoie les données des users et des transactions, puis les fusionne et les enregistre dans un fichier CSV.

    Args:
        users_path (str): Chemin vers le fichier des users.
        transactions_path (str): Chemin vers le fichier des transactions.
        output_path (str): Chemin vers le répertoire de sortie.

    Returns:
        pd.DataFrame: Le DataFrame fusionné et nettoyé.
    """
    # Assurez-vous que le répertoire de sortie existe
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Fonction pour charger les données des users
    def load_user_data(path: str) -> pd.DataFrame:
        """
        Charge les données des users depuis un fichier JSON.

        Args:
            path (str): Chemin vers le fichier JSON.

        Returns:
            pd.DataFrame: Données des users.
        """
        df = pd.read_json(path, dtype={"national_id": str})
        return df

    # Fonction pour charger les données des transactions
    def load_transaction_data(path: str) -> pd.DataFrame:
        """
        Charge les données des transactions depuis un fichier JSON.

        Args:
            path (str): Chemin vers le fichier JSON.

        Returns:
            pd.DataFrame: Données des transactions.
        """
        df = pd.read_json(path)
        return df

    # Fonction pour nettoyer les dates
    def clean_dates(value, output_format: str = "%d/%m/%Y %H:%M:%S") -> str | float:
        """
        Nettoie et formate les dates.

        Args:
            value: Valeur de la date.
            output_format (str): Format de sortie souhaité.

        Returns:
            str | float: Date formatée ou NaN si invalide.
        """
        if pd.isna(value):
            return np.nan
        value = str(value).strip()
        formats = ["%d-%m-%Y-%H-%M", "%Y/%m/%d %H:%M"]  # formats présents dans le fichier
        for fmt in formats:
            try:
                parsed = datetime.strptime(value, fmt)
                return parsed.strftime(output_format)
            except ValueError:
                continue
        return np.nan

    # Fonction pour convertir une chaîne en majuscules
    def correct_to_uppercase(string: str):
        """
        Convertit une chaîne en majuscules si elle ne l'est pas déjà.

        Args:
            string (str): Chaîne à convertir.

        Returns:
            str: Chaîne en majuscules.
        """
        return string.upper() if not string.isupper() else string

    # Fonction pour masquer les emails
    def hide_email(email: str) -> str:
        """
        Masque une partie de l'email pour des raisons de confidentialité.

        Args:
            email (str): Adresse email.

        Returns:
            str: Email masqué.
        """
        at_index = email.find("@")
        hidden_part = "*" * (at_index - 1)
        return email[0] + hidden_part + email[at_index:]

    # Fonction pour masquer les identifiants nationaux
    def hide_national_id(id: str) -> str:
        """
        Masque une partie de l'identifiant national pour des raisons de confidentialité.

        Args:
            id (str): Identifiant national.

        Returns:
            str: Identifiant masqué.
        """
        id = str(id)
        return id[:3] + "X" * len(id[3:])

    # Fonction pour aplatir les données de localisation
    def flatten_location(users: pd.DataFrame) -> pd.DataFrame:
        """
        Aplati les données de localisation imbriquées dans le DataFrame des users.

        Args:
            users (pd.DataFrame): Données des users.

        Returns:
            pd.DataFrame: Données des users avec colonnes aplaties.
        """
        location_data = pd.json_normalize(users["location"].to_list())
        users = users.drop(columns=["location"])
        users = pd.concat([users, location_data], axis=1)
        return users

    # Fonction pour nettoyer les données des users
    def clean_users(users: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données des users.

        Args:
            users (pd.DataFrame): Données des users.

        Returns:
            pd.DataFrame: Données des users nettoyées.
        """
        users = flatten_location(users)
        users = users.drop_duplicates()
        users["account_created"] = users["account_created"].apply(clean_dates)
        users["user_id"] = users["user_id"].apply(correct_to_uppercase)
        users["email"] = users["email"].apply(hide_email)
        users["national_id"] = users["national_id"].apply(hide_national_id)
        return users

    # Fonction pour nettoyer les données des transactions
    def clean_transactions(transactions: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données des transactions.

        Args:
            transactions (pd.DataFrame): Données des transactions.
            users (pd.DataFrame): Données des users.

        Returns:
            pd.DataFrame: Données des transactions nettoyées.
        """
        pd.set_option('display.float_format', '{:.2f}'.format)
        transactions.rename(columns={"tx_id": "ID"}, inplace=True)
        transactions["timestamp"] = transactions["timestamp"].apply(clean_dates)
        transactions["user_id"] = transactions["user_id"].apply(correct_to_uppercase)
        users_id_set = set(users["user_id"])
        transactions = transactions[transactions["user_id"].isin(users_id_set)]
        return transactions

    # Chargement des fichiers
    users: pd.DataFrame = load_user_data(users_path)
    transactions: pd.DataFrame = load_transaction_data(transactions_path)

    # Nettoyage des données
    users = clean_users(users)
    transactions = clean_transactions(transactions, users)

    # Fusion et sauvegarde dans un fichier CSV
    result: pd.DataFrame = users.merge(transactions, on="user_id", how="inner")
    result.to_csv(Path(output_path) / "output.csv", index=False)

    return result

if __name__ == "__main__":
    clean_data(DIRTY_USERS_FILE, DIRTY_TRANSACTIONS_FILE, OUTPUT_DIRECTORY)