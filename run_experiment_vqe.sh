source activate qiskit_aer

WORK_PATH=/home/jmlarkin/s2qpu/
#DATA_PATH=/home/jmlarkin/scratch/_experiment/
DATA_PATH=/pylon5/ci4s8dp/jmlarkin/_experiment/

if [ "$1" == "" ]; then
    echo "Positional parameter 1 is empty, needs to be experiment number"
else

    if [ "$1" == "1" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='linear' --vqe_depth=1 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out

    elif [ "$1" == "2" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='linear' --vqe_depth=2 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "3" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='linear' --vqe_depth=3 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "4" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='linear' --vqe_depth=4 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "5" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='linear' --vqe_depth=5 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        

    elif [ "$1" == "6" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='full' --vqe_depth=1 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out

    elif [ "$1" == "7" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='full' --vqe_depth=2 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "8" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='full' --vqe_depth=3 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "9" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='full' --vqe_depth=4 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "10" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='full' --vqe_depth=5 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out     



    elif [ "$1" == "11" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='linear' --vqe_depth=1 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out

    elif [ "$1" == "12" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='linear' --vqe_depth=2 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "13" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='linear' --vqe_depth=3 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "14" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='linear' --vqe_depth=4 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "15" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='linear' --vqe_depth=5 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        

    elif [ "$1" == "16" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='full' --vqe_depth=1 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out

    elif [ "$1" == "17" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='full' --vqe_depth=2 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "18" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='full' --vqe_depth=3 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "19" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='full' --vqe_depth=4 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "20" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='full' --vqe_depth=5 --vqe_optimizer='SPSA' --datapath=$2 &> $2.$1.vqe.out  
        

    elif [ "$1" == "21" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='linear' --vqe_depth=1 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out

    elif [ "$1" == "22" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='linear' --vqe_depth=2 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "23" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='linear' --vqe_depth=3 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "24" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='linear' --vqe_depth=4 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out        
    
    elif [ "$1" == "25" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='linear' --vqe_depth=5 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out    

    elif [ "$1" == "26" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='full' --vqe_depth=1 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out

    elif [ "$1" == "27" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='full' --vqe_depth=2 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "28" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='full' --vqe_depth=3 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "29" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='full' --vqe_depth=4 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "30" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RY' --vqe_entangler='full' --vqe_depth=5 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out     



    elif [ "$1" == "31" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='linear' --vqe_depth=1 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out

    elif [ "$1" == "32" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='linear' --vqe_depth=2 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "33" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='linear' --vqe_depth=3 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "34" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='linear' --vqe_depth=4 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "35" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='linear' --vqe_depth=5 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out        

    elif [ "$1" == "36" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='full' --vqe_depth=1 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out

    elif [ "$1" == "37" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='full' --vqe_depth=2 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "38" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='full' --vqe_depth=3 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "39" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='full' --vqe_depth=4 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out        
        
    elif [ "$1" == "40" ]; then
        python run_experiment.py --vqe_aer --vqe_sim --num_shots=1000 --vqe_var_form='RYRZ' --vqe_entangler='full' --vqe_depth=5 --vqe_optimizer='COBYLA' --datapath=$2 &> $2.$1.vqe.out  

    else
        echo "HI"
    fi

fi


