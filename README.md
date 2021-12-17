# renyiqnets
Quantum Generative Training with an Unbounded Loss Function

## Requirements:
Tensorflow 2.2.0, NumPy 1.18.1, QuTiP 4.5.1

## Unitary Quantum Network Training script: 

usage: unitary_train.py [-h] [--num_qb NUM_QB] [--vis_qb VIS_QB] \
                        [--units_2x UNITS_2X] [--lr LR] [--s_arc S_ARC] \
                        [--s_state S_STATE] [--s_init S_INIT] \
                        [--folder FOLDER]

example: python unitary_train.py --num_qb 5 --vis_qb 4 --units_2x 0 --lr 0.001 
 --s_arc 2 --s_state 4 --s_init 6 --folder ./loss/

optional arguments:\
  -h, --help &nbsp;&nbsp;           Show this help message. \
  --num_qb NUM_QB &nbsp; &nbsp;     Size of the total number of qubits i.e. visible+hidden \
  --vis_qb VIS_QB &nbsp; &nbsp;     Size of the number of visible qubits. \
  --units_2x UNITS_2X &nbsp; &nbsp; Twice as many units? \
  --lr LR &nbsp; &nbsp;             Learning rate. \
  --s_arc S_ARC &nbsp; &nbsp;       Random seed for neural architecture. \
  --s_state S_STATE &nbsp; &nbsp;   Random seed for rho (training data). \
  --s_init S_INIT &nbsp;&nbsp;      Random seed for network initialization. \
  --folder FOLDER &nbsp;&nbsp;      Folder to store loss and accuracy.

## Quantum Boltzmann Machine Training script
usage: bm_train.py [-h] [--num_qb NUM_QB] [--vis_qb VIS_QB] \
                   [--units_2x UNITS_2X] [--lr LR] [--l2_penalty L2_PENALTY] \
                   [--s_state S_STATE] [--s_init S_INIT] [--t_temp T_TEMP] \
                   [--epochs EPOCHS] [--folder FOLDER]

optional arguments:\
  -h, --help &nbsp;&nbsp;           Show this help message and exit \
  --num_qb NUM_QB &nbsp;&nbsp;      Size of the total number of qubits i.e. visible+hidden \
  --vis_qb VIS_QB &nbsp;&nbsp;      Size of the number of visible qubits. \
  --units_2x UNITS_2X &nbsp;&nbsp;  Twice as many units? \
  --lr LR &nbsp;&nbsp;              Learning rate. \
  --l2_penalty L2_PENALTY &nbsp; &nbsp; L2 regularization penalty. \
  --s_state S_STATE &nbsp;&nbsp;    Random seed for rho (training data). \
  --s_init S_INIT &nbsp;&nbsp;      Random seed for network initialization. \
  --t_temp T_TEMP &nbsp;&nbsp;      Target state temperture coeficient. \
  --epochs EPOCHS &nbsp;&nbsp;      Number of training epochs. \
  --folder FOLDER  &nbsp;&nbsp;     Folder to store loss and accuracy.
