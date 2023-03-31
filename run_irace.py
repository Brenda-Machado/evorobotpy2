"""
Script que chama o terminal R, roda o irace no background e armazena os valores printados no terminal
"""

import os
import subprocess
import sys
import time

# Path inicial do irace        
path = "/home/tuning"

# Path do arquivo de saída
path_output = "/home/irace_output.txt"

# Função que chama o terminal R e roda o irace no background
def run_irace():
    # Chama o terminal R
    subprocess.call(["R"])
    # Entra no diretório do irace
    os.chdir(path)
    # Roda o irace no background e armazena os valores printados no terminal
    subprocess.call(["R", "-e", "library(irace); scenario <- readScenario(filename = 'scenario.txt', scenario = defaultScenario()); irace.main(scenario = scenario); irace(\"/home/irace_config.txt\")"], stdout=open(path_output, "w"))

run_irace()