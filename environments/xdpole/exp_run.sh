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
e_folder="/root/evorobotpy2/environments/xdpole"

# Which .ini file to use?
ini_name="/root/evorobotpy2/environments/xdpole/ErDpole.ini"

# Paramters to change in the .ini file
sample=$sample
episode=$episodes

# Change the .ini file
#sed -i "s/sampleSize = .*/sampleSize = $sample/g" $ini_name
#sed -i "s/episodes = .*/episodes = $episode/g" $ini_name

# How many seeds do you want to run for each experiment?
number_of_seeds=5

#How many experiments do you want to run simultaneously?
simultaneously=5

#Seed initial number
seeds_numbers=()

total_of_processes=$number_of_seeds

commands_list=()

seed=$((RANDOM%100+1))

seeds_numbers+=("${seed}")

for n in $(seq 1 $number_of_seeds)
do
    find_folder="cd $e_folder"
    seed_command="python3 /root/evorobotpy2/bin/es.py -f $ini_name"
    seed_command="$seed_command -s $seed"
    new_command=("$find_folder" "$seed_command")
    commands_list+=("${new_command[@]}")
    # Generate new seed if it's already in the array
    while true
    do
        found=false
        for s in "${seeds_numbers[@]}"
        do
            if [ $s -eq $seed ]
            then
                found=true
                break
            fi
        done
        if $found
        then
            seed=$((RANDOM%100+1))
        else
            seeds_numbers+=("${seed}")
            break
        fi
    done
done

function run_process {
    eval $1
}

for i in "${commands_list[@]}"
do
    run_process "$i" &
done

wait
