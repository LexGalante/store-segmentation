# -*- coding: utf-8 -*-
"""
Funcões auxiliares para o projeto
"""

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