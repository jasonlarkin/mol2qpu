{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cvxpy as cvx\n",
    "import cvxopt\n",
    "import networkx as nx\n",
    "import numpy as np\n",
    "import argparse\n",
    "from typing import Tuple\n",
    "# %matplotlib inline\n",
    "import matplotlib.pyplot as plt\n",
    "import random\n",
    "from time import time\n",
    "import os, sys\n",
    "import subprocess\n",
    "import re\n",
    "import numpy as np\n",
    "import pylab\n",
    "from qiskit import Aer, BasicAer\n",
    "from qiskit.aqua import aqua_globals, QuantumInstance\n",
    "from qiskit.aqua.algorithms import ExactEigensolver, VQE\n",
    "from qiskit.aqua.components.optimizers import SPSA, COBYLA, L_BFGS_B\n",
    "from qiskit.aqua.components.variational_forms import RY, RYRZ\n",
    "from qiskit.chemistry.drivers import PySCFDriver, UnitsType\n",
    "from qiskit.chemistry.core import Hamiltonian, QubitMappingType, TransformationType"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import types\n",
    "args = types.SimpleNamespace()\n",
    "\n",
    "args.molecule='Be2' \n",
    "args.basis_set='ccpvtz' \n",
    "args.algorithm='VQE' \n",
    "args.transformation_type='TransformationType.FULL' \n",
    "args.qubitmapping_type='QubitMappingType.BRAVYI_KITAEV' \n",
    "args.two_qubit_reduce = True\n",
    "args.vqe_optimizer='SPSA' \n",
    "args.vqe_max_iter=2\n",
    "args.vqe_var_form='RY' \n",
    "args.vqe_depth=1 \n",
    "args.vqe_entangler='linear' \n",
    "args.num_shots=100 \n",
    "args.max_parallel_threads=10 \n",
    "args.vqe_sim = True\n",
    "args.vqe_aer = True\n",
    "args.datapath='/pylon5/cc5phsp/cbernaci/mol2qpu'\n",
    "args.random_seed=750 \n",
    "args.vqe_opt_params = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "BEGIN RUN_VQE_AER\n",
      "args.molecule= Be2\n",
      "args.datapath= /pylon5/cc5phsp/cbernaci/mol2qpu\n"
     ]
    }
   ],
   "source": [
    "#def run_vqe_aer(args):\n",
    "\n",
    "print(\"BEGIN RUN_VQE_AER\")\n",
    "print(\"args.molecule=\", args.molecule)\n",
    "print(\"args.datapath=\", args.datapath)\n",
    "#sys.exit(\"exiting now\")\n",
    "\n",
    "### READ IN MOLECULE\n",
    "#FIXME: bring in molecules from file?\n",
    "if args.molecule=='H2':\n",
    "    molecule = 'H .0 .0 -{0}; H .0 .0 {0}'\n",
    "elif args.molecule=='LiH':\n",
    "    molecule = 'Li .0 .0 -{0}; H .0 .0 {0}'\n",
    "elif args.molecule=='Be2':\n",
    "    molecule = 'Be .0 .0 -{0}; Be .0 .0 {0}'\n",
    "\n",
    "start = 0.5  # Start distance\n",
    "steps = 0    # Number of steps to increase by\n",
    "energies = np.zeros(steps+1)\n",
    "hf_energies = np.zeros(steps+1)\n",
    "distances = np.zeros(steps+1)\n",
    "aqua_globals.random_seed = args.random_seed\n",
    "\n",
    "d = start\n",
    "\n",
    "driver = PySCFDriver(molecule.format(d/2), basis=args.basis_set)\n",
    "qmolecule = driver.run()\n",
    "operator =  Hamiltonian(transformation=eval(args.transformation_type), \n",
    "                        qubit_mapping=eval(args.qubitmapping_type),  \n",
    "                        two_qubit_reduction=args.two_qubit_reduce)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "qubitOp, aux_ops = operator.run(qmolecule)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "backend = Aer.get_backend('qasm_simulator')\n",
    "\n",
    "quantum_instance = QuantumInstance(circuit_caching=True, \n",
    "                                   backend=backend,\n",
    "                                   backend_options={'max_parallel_threads': args.max_parallel_threads,                                                                             'max_parallel_experiments': 0, \n",
    "                                                    'shots': args.num_shots})\n",
    "## optimizer\n",
    "if args.vqe_optimizer=='SPSA':\n",
    "    optimizer = SPSA(max_trials=200)\n",
    "elif args.vqe_optimizer=='COBYLA':\n",
    "    optimizer = COBYLA()\n",
    "    optimizer.set_options(maxiter=args.vqe_max_iter)\n",
    "elif args.vqe_optimizer=='L_BFGS_B':\n",
    "    optimizer = L_BFGS_B(maxfun=args.vqe_max_iter)\n",
    "else:\n",
    "    optimizer = COBYLA()\n",
    "    optimizer.set_options(maxiter=args.vqe_max_iter)\n",
    "\n",
    "## variational form\n",
    "if args.vqe_var_form=='RY':\n",
    "    var_form = RY(qubitOp.num_qubits, depth=args.vqe_depth, entanglement=args.vqe_entangler)   \n",
    "elif args.vqe_var_form=='RYRZ':\n",
    "    var_form = RYRZ(qubitOp.num_qubits, depth=args.vqe_depth, entanglement=args.vqe_entangler)\n",
    "\n",
    "## VQE params\n",
    "if args.vqe_opt_params:\n",
    "    initial_point=np.load(args.vqe_opt_params_path+'._ret_opt_params.npy',allow_pickle=True, fix_imports=True)\n",
    "    algo = VQE(qubitOp, var_form, optimizer, initial_point=initial_point)\n",
    "else:\n",
    "    algo = VQE(qubitOp, var_form, optimizer)\n",
    "\n",
    "result = algo.run(quantum_instance)    \n",
    "lines, result_op = operator.process_algorithm_result(result)\n",
    "\n",
    "energies[i] = result['energy']\n",
    "hf_energies[i] = result['hf_energy']\n",
    "distances[i] = d\n",
    "\n",
    "#print(type(result))\n",
    "#print(dir(algo))\n",
    "circ_opt = algo.get_optimal_circuit()\n",
    "print(circ_opt)\n",
    "#print('vqe WallTime:',computeWalltime(circ_opt))\n",
    "print(' --- complete')\n",
    "print('circuit_summary=', quantum_instance.circuit_summary)\n",
    "print(algo.print_settings())\n",
    "print('Distances: ', distances)\n",
    "print('Energies:', energies)\n",
    "print('Hartree-Fock energies:', hf_energies)\n",
    "\n",
    "#return result\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([{'10': 9, '01': 91}], dtype=object)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result['algorithm_retvals']['eigvecs']\n",
    "#results['eigvecs']\n",
    "#type(results)\n",
    "#eigvecs = results.get('eigvecs')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 504x360 with 1 Axes>"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from qiskit.tools.visualization import plot_histogram\n",
    "plot_histogram(results['algorithm_retvals']['eigvecs'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'operator' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-34-8ea413b4d767>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mlines\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mresultA\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0moperator\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mprocess_algorithm_result\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mresults\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m: name 'operator' is not defined"
     ]
    }
   ],
   "source": [
    "lines, resultA = operator.process_algorithm_result(results)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#results = run_vqe_aer(args) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}