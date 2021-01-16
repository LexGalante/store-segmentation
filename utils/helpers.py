# -*- coding: utf-8 -*-
"""
Funcões auxiliares para o projeto
"""
import numpy as np
import pandas as pd
from typing import Any


def handle_boolean(input: str, default: int = 0) -> int:
    """
    Transforma os valores booleanos (S, N)
    N -> 0
    S -> 1
    """
    if input == "S":
        return 1
    elif input == "N":
        return 0
    else:
        return default

def handle_club(input: str) -> int:
    """
    Transforma os valores dos club
    DIAMANTE -> 1
    OURO -> 2
    PRATA -> 3
    BRONZE -> 4
    """
    if input == "DIAMANTE":
        return 1
    elif input == "OURO":
        return 2
    elif input == "PRATA":
        return 3
    elif input == "BRONZE":
        return 4
    
    raise ValueError(f"Club ({input}) desconhecido, verifique")


def handle_number_of_employees(cols: list):
    number_of_employess = cols[0]
    club = cols[1]
    
    if pd.isna(number_of_employess):
        if club == "DIAMANTE":
            return 19
        elif club == "OURO":
            return 11
        elif club == "PRATA":
            return 7
        elif club == "BRONZE":
            return 18
    
    return number_of_employess
    
    
def handle_nan_float(input: Any) -> Any:
    """
    Transforma os valores nulos no parametro value
    """
    if pd.isna(input):
        return 0
    
    return input

    
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

