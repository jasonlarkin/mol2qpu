#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM
#SBATCH -t 01:00:00
#SBATCH --ntasks-per-node 28

#source activate qiskit_aer
source activate qiskit-0.14

WORK_PATH=/home/jmlarkin/mol2qpu/
DATA_PATH=/pylon5/ci4s8dp/jmlarkin/_experiment/mol2qpu

if [ "$1" == "" ]; then
    echo "Positional parameter 1 is empty, needs to be experiment number"
else
    if [ "$1" == "1" ]; then    
       echo "HI, you have chosen...wisely."   
       #echo $DATA_PATH/$1.H2.vqe.out
       python run_experiment.py --molecule='H2' \ #--random_seed=750 ## throws error in numpy if u leave this in
       --basis_set='sto3g' --algorithm='VQE' \
       --transformation_type='TransformationType.FULL' --qubitmapping_type='QubitMappingType.PARITY' --two_qubit_reduce \
       --vqe_optimizer='SPSA' --vqe_max_iter=10 --vqe_var_form='RY' --vqe_depth=3 --vqe_entangler='linear' --num_shots=1000 \
       --max_parallel_threads=10 --vqe_sim --vqe_aer --datapath=$DATA_PATH &> $DATA_PATH/$1.H2.vqe.out
       #--vqe_optimizer=='COBYLA'
       #--vqe_optimizer=='L_BFGS_B'
       #--vqe_var_form=='RYRZ'      
       #--vqe_entangler=='full'                          
       #MOL_PATH="$DATA_PATH/molecules/H2.txt" 
       
    elif [ "$1" == "2" ]; then
       echo "HI"
       #MOL_PATH="$DATA_PATH/molecules/H2.txt"  
       python run_experiment.py --molecule='H2' \
       #--random_seed==750' ## throws error in numpy if u leave this in
       --basis_set=='sto3g'--algorithm='ExactEigensolver' \
       --transformation_type=='TransformationType.FULL' --qubitmapping_type=='QubitMappingType.PARITY' \
       --two_qubit_reduce --vqe_optimizer=='SPSA' --vqe_max_iter==10--vqe_var_form=='RY' --vqe_depth==3 \
       --vqe_entangler=='linear' --num_shots==1000 --max_parallel_threads==10 --vqe_sim --vqe_aer \
       --datapath=="$DATA_PATH/output" &> $1.LiH.vqe.out  
    else
        echo "HI"
    fi
fi


