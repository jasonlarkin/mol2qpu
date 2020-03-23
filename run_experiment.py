import cvxpy as cvx
import cvxopt
import networkx as nx
import numpy as np
import argparse

from typing import Tuple

# %matplotlib inline
import matplotlib.pyplot as plt

import random
from time import time
import os, sys
import subprocess
import re



def parse_args():    
    print("BEGIN PARSING ARGS")
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='_experiment/mol2qpu')
    parser.add_argument('--outpath', type=str, default='_experiment/')

    parser.add_argument('--molecule', type=str, default='H2')
    parser.add_argument('--random_seed', type=float, default=750)
    parser.add_argument('--basis_set', type=str, default='sto3g')
    parser.add_argument('--algorithm', type=str, default='VQE')
    parser.add_argument('--transformation_type', type=str, default='TransformationType.FULL')
    parser.add_argument('--qubitmapping_type', type=str, default='QubitMappingType.PARITY')
    parser.add_argument('--two_qubit_reduce', action='store_true')    
    parser.add_argument('--vqe_optimizer', type=str, default='SPSA')
    parser.add_argument('--vqe_max_iter', type=int, default=250)
    parser.add_argument('--vqe_var_form', type=str, default='RY')
    parser.add_argument('--vqe_depth', type=int, default=1)
    parser.add_argument('--vqe_entangler', type=str, default='linear')
    parser.add_argument('--max_parallel_threads', type=int, default=28)
    parser.add_argument('--num_shots', type=int, default=100)
    parser.add_argument('--vqe_aer', action='store_true') #run aer
    parser.add_argument('--vqe_sim', action='store_true') #run aer
    parser.add_argument('--vqe_opt_params', action='store_true') #use initial
    parser.add_argument('--vqe_opt_params_path', type=str, default='_experiment') #use initial
    
    parser.add_argument('--T1', type=float, default=200)
    parser.add_argument('--T2', type=float, default=200)
    parser.add_argument('--L1', type=int, default=2)
    parser.add_argument('--L2', type=int, default=2)
    print("DONE PARSING ARGS")
    return parser.parse_args()

def run_vqe_aer():
    
    import numpy as np
    import pylab
    from qiskit import Aer, BasicAer
    from qiskit.aqua import aqua_globals, QuantumInstance
    from qiskit.aqua.algorithms import ExactEigensolver, VQE
    from qiskit.aqua.components.optimizers import SPSA, COBYLA, L_BFGS_B
    from qiskit.aqua.components.variational_forms import RY, RYRZ
    from qiskit.chemistry.drivers import PySCFDriver, UnitsType
    from qiskit.chemistry.core import Hamiltonian, QubitMappingType, TransformationType
    
    
    ### 1. FROM JASONS ORIGINAL: IMPORTS
    #import networkx as nx
    #from qiskit.tools.visualization import plot_histogram
    #from qiskit.aqua import Operator, run_algorithm
    #from qiskit.aqua.input import EnergyInput
    #from qiskit.aqua.translators.ising import max_cut, tsp
       
    
    ### 2. FROM JASONS ORIGINAL: BRING IN GRAPH
    #G = nx.read_edgelist(path=args.datapath)
    #print('nnode=', G.number_of_nodes())
    #print('nedges=',G.edges())
    #nqbits=G.number_of_nodes()
    ##G.add_weighted_edges_from(G.edges())
    #w = nx.to_numpy_array(G,dtype=np.int32)    
    #print(w)
    
    ### 3. FROM JASONS ORIGINAL: CONSTRUCT MAPPING
    #qubitOp, offset = max_cut.get_max_cut_qubitops(w)
    #algo_input = EnergyInput(qubitOp)
    
    print("BEGIN RUN_VQE_AER")
    print("args.molecule=", args.molecule)
    print("args.datapath=", args.datapath)
    #sys.exit("exiting now")
   
    ### READ IN MOLECULE
    #FIXME: bring in molecules from file?
    if args.molecule=='H2':
        molecule = 'H .0 .0 -{0}; H .0 .0 {0}'
    elif args.molecule=='LiH':
        molecule = 'Li .0 .0 -{0}; H .0 .0 {0}'
        
    start = 0.5  # Start distance
    by    = 0.5  # How much to increase distance by
    steps = 1  # Number of steps to increase by
    energies = np.zeros(steps+1)
    hf_energies = np.zeros(steps+1)
    distances = np.zeros(steps+1)
    aqua_globals.random_seed = args.random_seed
    
    
    ### RUN OUTERLOOP: PERFORM VQE FOR EACH ENERGY
    for i in range(steps+1):
    
        d = start + i*by/steps
        print("i = ", i)
        print("d = ", d)
        
        driver = PySCFDriver(molecule.format(d/2), basis=args.basis_set)
        
        qmolecule = driver.run()
        operator =  Hamiltonian(transformation=eval(args.transformation_type), 
                                qubit_mapping=eval(args.qubitmapping_type),  
                                two_qubit_reduction=args.two_qubit_reduce)
        
        qubitOp, aux_ops = operator.run(qmolecule)
        
        if args.algorithm == 'VQE':
            
            if args.vqe_sim:
                backend = Aer.get_backend('qasm_simulator')
                
                quantum_instance = QuantumInstance(circuit_caching=True, 
                                                   backend=backend,
                                                   backend_options={'max_parallel_threads': args.max_parallel_threads,                                                                             'max_parallel_experiments': 0, 
                                                                    'shots': args.num_shots})
            
            ## optimizer
            if args.vqe_optimizer=='SPSA':
                optimizer = SPSA(max_trials=200)
            elif args.vqe_optimizer=='COBYLA':
                optimizer = COBYLA()
                optimizer.set_options(maxiter=args.vqe_max_iter)
            elif args.vqe_optimizer=='L_BFGS_B':
                optimizer = L_BFGS_B(maxfun=args.vqe_max_iter)
            else:
                optimizer = COBYLA()
                optimizer.set_options(maxiter=args.vqe_max_iter)
            
            ## variational form
            if args.vqe_var_form=='RY':
                var_form = RY(qubitOp.num_qubits, depth=args.vqe_depth, entanglement=args.vqe_entangler)   
            elif args.vqe_var_form=='RYRZ':
                var_form = RYRZ(qubitOp.num_qubits, depth=args.vqe_depth, entanglement=args.vqe_entangler)
            
            
            ## VQE params
            if args.vqe_opt_params:
                initial_point=np.load(args.vqe_opt_params_path+'._ret_opt_params.npy',allow_pickle=True, fix_imports=True)
                algo = VQE(qubitOp, var_form, optimizer, initial_point=initial_point)
            else:
                algo = VQE(qubitOp, var_form, optimizer)
            
            result = algo.run(quantum_instance)
            
        elif args.algorithm == 'ExactEigensolver':
            result = ExactEigensolver(qubit_op).run()            
                      
        lines, result = operator.process_algorithm_result(result)
        energies[i] = result['energy']
        hf_energies[i] = result['hf_energy']
        distances[i] = d
    
    #print(type(result))
    #print(dir(algo))
    circ_opt = algo.get_optimal_circuit()
    print(circ_opt)
    #print('vqe WallTime:',computeWalltime(circ_opt))
    print(' --- complete')
    print('circuit_summary=', quantum_instance.circuit_summary)
    print(algo.print_settings())
    print('Distances: ', distances)
    print('Energies:', energies)
    print('Hartree-Fock energies:', hf_energies)
            
    #fig = pylab.plot(distances, hf_energies, label='Hartree-Fock')
    #for j in range(len(algorithms)):
    #    fig.pylab.plot(distances, energies[j], label=algorithms[j])
    #    fig.pylab.xlabel('Interatomic distance (Angstrom)')
    #    fig.pylab.ylabel('Energy (Hartree)')
    #    fig.pylab.title('H2 Ground State Energy')
    #    fig.pylab.legend(loc='upper right');
    #    fig.savefig(args.outpath+'.counts.png', format='PNG') 
        
    #print('vqe WallTime:',computeWalltime(quantum_instance,result))   
        
    #x = max_cut.sample_most_likely(result['eigvecs'][0])
    #print('energy:', result['energy'])
    #print('time:', result['eval_time'])
    #print('opt count:', result['eval_count'])
    #print('maxcut objective:', result['energy'] + offset)
    #print('solution:', max_cut.get_graph_solution(x))
    #print('solution objective:', max_cut.max_cut_value(x, w))
    
    #colors = ['r' if max_cut.get_graph_solution(x)[i] == 0 else 'b' for i in range(nqbits)]
    #pos = nx.spring_layout(G)
    #default_axes = plt.axes(frameon=True)
    #nx.draw_networkx(G, node_color=colors, node_size=600, alpha = .8, pos=pos)
    #plt.savefig(args.outpath+'.result.png', format='PNG')
    #plt.clf()
    
    #if args.vqe_sim:
    #    fig = plot_histogram(result['eigvecs'][0])
    #    ax = fig.axes[0]
    #    ax.set_ylim(0, 1)
    #    fig.savefig(args.outpath+'.counts.png', format='PNG')    
    #elif args.vqe_sim_sv:
    #    print(dir(result))
    #    print(max_cut.sample_most_likely(result['eigvecs'][0]))
    
    
    
       
def computeWalltime(quantum_instance,result):
    circ_cache = quantum_instance.circuit_cache
    
    
    
#    print('circ_cache=',circ_cache)
    experiments_dict=circ_cache.qobjs[0].as_dict()
    experiments=experiments_dict['experiments']
    
    gate_times_small={}
    gate_times_small['u1']=100
    gate_times_small['u2']=100
    gate_times_small['cx']=200
    gate_times_small['measure']=500

#    print('experiments[0]=',experiments)
    
    qaoa_circ_wall_time=0
    for inst in experiments[0]['instructions']:
#        print('inst=',inst['name'])
        if inst['name']=='u1':
#            print('u1=',inst)
            qaoa_circ_wall_time+=gate_times_small[inst['name']]
        elif inst['name']=='u2':
#            print('u2=',inst)    
            qaoa_circ_wall_time+=gate_times_small[inst['name']]
        elif inst['name']=='cx':
#            print('cx=',inst)
#            print('qubits=',inst['qubits'])        
            qaoa_circ_wall_time+=gate_times_small[inst['name']]
        elif inst['name']=='measure':
#            print('cx=',inst)
#            print('qubits=',inst['qubits'])        
            qaoa_circ_wall_time+=gate_times_small[inst['name']]

    qaoa_wall_time=result['eval_count']*qaoa_circ_wall_time/1e3
            
    print('qaoa_wall_time (ms)=',result['eval_count']*qaoa_circ_wall_time/1e3)
    
    return qaoa_wall_time
        
def BuildQPUNoiseModel(L1,L2,T1,T2):

    from qiskit.providers.aer import noise
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.providers.aer.noise.errors import depolarizing_error

    import networkx as nx

    #create custom coupl_map of L1xL2=num_qubit device
    G = nx.grid_2d_graph(L1,L2)
#    nx.draw(G,node_size=0.05)
#    plt.show()
 
    coupling_map=[]
    for edge in G.edges():
        print('edge[0][0]=',edge[0][0],'edge[0][1]=',edge[0][1],'edge[1][0]=',edge[1][0],'edge[1][1]=',edge[1][1])
        nodei=edge[0][0]*L1 + edge[0][1]
        print('nodei=',nodei)
        nodej=edge[1][0]*L1 + edge[1][1]
        print('nodej=',nodej)
        coupling_map.append([nodei,nodej])
        
    gate_times = [
        ('u1', None, 0), ('u2', None, 100), ('u3', None, 200)]
    gate_time=100 #ns?
    for pair in coupling_map:
        gate_times.append(('cx',pair,gate_time))
    gate_times.append(('cx', None, 200))
        
    from qiskit.providers.aer.noise.errors import thermal_relaxation_error

    num_qubits=L1*L2

    # T1 and T2 values for num_qubits
    #updated with Gian Intel MaxCut paper values roughly
    T1s = np.random.normal(T1*1e3, 10e3, num_qubits) # Sampled from normal distribution mean 250 microsec
    T2s = np.random.normal(T2*1e3, 10e3, num_qubits)  # Sampled from normal distribution mean 250 microsec

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(num_qubits)])

    # Instruction times (in nanoseconds)
    time_u1 = 0   # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100 # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000 # 1 microsecond

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                  for t1, t2 in zip(T1s, T2s)]
    errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
              for t1, t2 in zip(T1s, T2s)]
    errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
              for t1, t2 in zip(T1s, T2s)]
    errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
              for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
             thermal_relaxation_error(t1b, t2b, time_cx))
              for t1a, t2a in zip(T1s, T2s)]
               for t1b, t2b in zip(T1s, T2s)]

    # Add errors to noise model
    noise_thermal = NoiseModel()

    for i in range(0, num_qubits):
        noise_thermal.add_quantum_error(errors_reset[i], "reset", [i])
        noise_thermal.add_quantum_error(errors_measure[i], "measure", [i])
        noise_thermal.add_quantum_error(errors_u1[i], "u1", [i])
        noise_thermal.add_quantum_error(errors_u2[i], "u2", [i])
        noise_thermal.add_quantum_error(errors_u3[i], "u3", [i])

    for pair in coupling_map:
#    print('pair[0]=',pair[0],'pair[1]=',pair[1])
#    print('errors_cx[pair[0]][pair[1]]=',errors_cx[pair[0]][pair[1]])
        noise_thermal.add_quantum_error(errors_cx[pair[0]][pair[1]], "cx", [pair[0], pair[1]])
    
    print(noise_thermal)

    
    return noise_thermal, coupling_map, gate_times


    
if __name__ == "__main__":
    print("Here we go")
    args = parse_args()
#    if args.create_mol == True:
#        os.makedirs(args.datapath, exist_ok=True)
#        make_data()
    print("vqe_aer = ", args.vqe_aer)
    if args.vqe_aer == True:
        run_vqe_aer()        
        
    print("We done here. Go Home.")
        

    
