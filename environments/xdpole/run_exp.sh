#!/bin/bash

# Get the arguments
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        --episodes)
        episodes="$2"
        shift # past argument
        shift # past value
        ;;
        --sample)
        sample="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        shift # past argument
        ;;
    esac
done

# Experiment folder (relative path)
e_folder="/home/arthur/code/evorobotpy2/environments/xdpole"

# Which .ini file to use?
ini_name="/home/arthur/code/evorobotpy2/environments/xdpole/ErDpole.ini"

# Paramters to change in the .ini file 
sample=$sample
episode=$episodes

# Change the .ini file
sed -i "s/sampleSize = .*/sampleSize = $sample/g" $ini_name
sed -i "s/episodes = .*/episodes = $episode/g" $ini_name

# How many seeds do you want to run for each experiment?
number_of_seeds=1

#How many experiments do you want to run simultaneously?
simultaneously=1

#Seed initial number
initial_seed_number=1

total_of_processes=$number_of_seeds

commands_list=()

seed=$initial_seed_number

for n in $(seq 1 $number_of_seeds)
do
    find_folder="cd $e_folder"
    seed_command="python3 ~/code/evorobotpy2/bin/es.py -f $ini_name"
    seed_command="$seed_command -s $seed"
    new_command=("$find_folder" "$seed_command")
    commands_list+=("${new_command[@]}")
    seed=$((seed+1))
done

function run_process {
    eval $1
}

for i in "${commands_list[@]}"
do
    run_process "$i" &
done

wait
