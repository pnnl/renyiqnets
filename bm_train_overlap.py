"""
Title: Training of a Quantum Boltzmann Machine using Fidelity

"""

import os
import sys
import csv
import argparse
import numpy as np
from functools import reduce
from operator import mul
from itertools import product
from random import seed as rs
from random import sample
from numpy.linalg import norm, multi_dot
from numpy.random import normal as N
from numpy.random import rand, randn, randint
from numpy.random import seed as nps
from numpy import zeros
from math import factorial
from qutip import Qobj
from qutip.random_objects import rand_unitary_haar, rand_herm, rand_dm
from qutip.operators import sigmax, sigmay, sigmaz, identity
from qutip.tensor import tensor
from qutip.states import maximally_mixed_dm, ket2dm, basis, thermal_dm
from qutip.metrics import tracedist, fidelity

import tensorflow as tf

#target state
from target_states import three_body_ts

#tf-multi
def multidot_tf(matrices):
    mat_tf = tf.stack(matrices)
    return tf.scan(lambda a, b: tf.matmul(a, b), mat_tf)[-1]

#Pytorch matrix-matrix multiplication
# split adjacent elements into separate tensors
def functools_reduce(x):
    return reduce(torch.matmul, x)
#def recursive_reduce_with_tensor(x):
#    N, M, _ = x.size()
#    while N > 1:
#        x = x.view(N//2, 2, M, M).permute(1,0,2,3)
#        x = torch.matmul(x[0], x[1])
#        N, M, _ = x.size()
#    return x.view(M,M)

def QRenyiDiv(sig, rho):
    return np.log( ((sig*sig)*rho.inv() ).tr())


def tensor_sigmas(positions, paulis, size):
    Is=[identity(2) for _ in range(size-len(positions))]
    for p, pauli in zip(positions, paulis):
        Is.insert(p, pauli)
    return tensor(Is)

class bm_model:
        def __init__(self, size, vis_size, rho, units_2x=False, no_y=False):
            self.size = size
            self.rho = rho
            self.vis_size = vis_size
            self.vis_registers = list(range(vis_size))
            self.rho_inv = rho.inv()
            self.units_2x = units_2x

            loop_z=[tensor_sigmas([i], [sigmaz()], size=size) 
                    for i in range(size)]
            loop_y=[tensor_sigmas([i], [sigmay()], size=size) 
                    for i in range(size)]
            loop_x=[tensor_sigmas([i], [sigmax()], size=size) 
                    for i in range(size)]

            offdiag=[(i,j) for i, j in list(product(range(size), range(size))) 
                     if i<j]

            inter_x=[tensor_sigmas([i,j], [sigmax(),sigmax()],size=size) for
                     i, j in offdiag]
            inter_y=[tensor_sigmas([i,j], [sigmay(),sigmay()],size=size) for
                     i, j in offdiag]
            inter_z=[tensor_sigmas([i,j], [sigmaz(),sigmaz()],size=size) for
                     i, j in offdiag]
            inter_xy=[tensor_sigmas([i,j], [sigmax(),sigmay()],size=size) for
                     i, j in offdiag]
            inter_xz=[tensor_sigmas([i,j], [sigmax(),sigmaz()],size=size) for
                     i, j in offdiag]
            inter_yz=[tensor_sigmas([i,j], [sigmax(),sigmaz()],size=size) for
                     i, j in offdiag]
            inter_yx=[tensor_sigmas([i,j], [sigmay(),sigmax()],size=size) for
                     i, j in offdiag]
            inter_zx=[tensor_sigmas([i,j], [sigmaz(),sigmax()],size=size) for
                     i, j in offdiag]
            inter_zy=[tensor_sigmas([i,j], [sigmaz(),sigmay()],size=size) for
                         i, j in offdiag]

            self.pauli_ops = loop_x+loop_y+loop_z+inter_x+inter_y+\
                                         inter_z+inter_xy+inter_xz+inter_yz+\
                                         inter_yx+inter_zx+inter_zy

            if no_y == True:
                self.pauli_ops = loop_x+loop_z+inter_x+inter_z+inter_xz+inter_zx

            if units_2x == True:
                self.pauli_ops = self.pauli_ops+loop_x+loop_y+\
                                             loop_z+inter_x+inter_y+inter_z+\
                                             inter_xy+inter_xz+inter_yz+\
                                             inter_yx+inter_zx+inter_zy

            self.num_params = len(self.pauli_ops)


            # randomly initialized, but normalized to prevent numerical
            # instabilities matrix inversion
            init_weights = randn(self.num_params)
            prod_terms = [(w*p) for w,p in 
                          zip(init_weights, self.pauli_ops)]
            H = -1*(sum(prod_terms))
            H_opnorm = norm(H, ord=2)
            self.weights = (1/H_opnorm)*init_weights


        def forward(self):
            prod_terms = [(w*p) for w,p in 
                          zip(self.weights, self.pauli_ops)]
            expH = ((-1)*sum(prod_terms)).expm()
            norm_expH = expH / expH.tr()
            return norm_expH

        def vis_forward(self):
            if self.vis_size == self.size:
                sigma_v = self.forward()
            else:
                sigma_v = self.forward().ptrace(self.vis_registers)
            return sigma_v

        def grad(self, k, h_tol=0.0001):
            prod_terms = [(w*p_op) for w, p_op in 
                          zip(self.weights, self.pauli_ops)]
            prod_terms[k] = ( (self.weights[k]+h_tol)*self.pauli_ops[k])
            H_htol = ((-1)*sum(prod_terms))
            H_htol_exp = H_htol.expm()
            norm_expH_htol = (H_htol_exp / H_htol_exp.tr()).ptrace(self.vis_registers)
            overlap = (self.vis_forward() * self.rho).tr()
            num_grad = ( (self.rho *norm_expH_htol).tr() - (overlap))/h_tol
            return num_grad

        def backward(self, l2_reg_penalty = 0.0):
            updates = np.array([self.grad(k)+l2_reg_penalty*2*w for k,w in 
                     zip(range(len(self.weights)), self.weights)])
            #check if gradients are real.
            if not np.all(abs(updates.imag) < 1e-14):
                print("Warning: Gradients contain a non-zero complex part:")
                print(updates.imag)
            print(' ')
            print('Max grad norm: ', np.abs(np.array(updates).real).max())
            print(' ')
            return np.array(updates).real


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_qb', type=int, default=None,
                        help='Size of the total number of qubits i.e. '+\
                        'visible+hidden')
    parser.add_argument('--vis_qb', type=int, default=None,
                        help='Size of the number of visible qubits.')
    parser.add_argument('--units_2x', type=int,
                        default=0,
                        help='Twice as many units?')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--l2_penalty', type=float, default=0,
                        help='L2 regularization penalty.')
    parser.add_argument('--s_state', type=int, default=7, 
                        help='Random seed for rho (training data).')
    parser.add_argument('--s_init', type=int, default=11, 
                        help='Random seed for network initialization.')
    parser.add_argument('--t_temp', type=float, default=5, 
                        help='Target state temperture coeficient.')
    parser.add_argument('--epochs', type=int, default=1000, 
                        help='Number of training epochs.')
    parser.add_argument('--folder', type=str, default='./', 
                        help='Folder to store loss and accuracy.')


    args = parser.parse_args()
    vis_qb = args.vis_qb
    num_qb = args.num_qb
    units_2x = args.units_2x
    s_state = args.s_state
    s_init = args.s_init
    lr=args.lr
    l2_reg = args.l2_penalty
    t_temp = args.t_temp
    epochs = args.epochs
    filename = args.folder+f'summary_overlap_num_{num_qb}_vis_{vis_qb}_2x_{units_2x}_'+\
    f'ss_{s_state}_si_{s_init}_lr_{lr}_l2_{l2_reg}_tstemp_{t_temp}_bm.csv'

    nps(s_state) #fix state
    rho = three_body_ts(vis_qb, temp=t_temp)
#    rho = rand_dm(2**vis_qb, dims=[vis_qb*[2],vis_qb*[2]])
#    rho = two_body_density(0.1, size=vis_qb, normalize=True, temp=t_temp)
    model = bm_model(num_qb, vis_qb, rho, units_2x=units_2x)

    print('Number of model parameters...', model.num_params)

    #set initial weights
    nps(s_init)
#    model.weights = 0.001*randn(model.num_params)
#    model.weights = 0.01+np.zeros(model.num_params)

    sigma_v = model.vis_forward()

    loss = QRenyiDiv(sigma_v, rho)
    tdist = tracedist(sigma_v, rho)
    fid = fidelity(sigma_v, rho)
    over = rho.overlap(sigma_v)


    f = open(filename, "a")
    writer = csv.DictWriter(f, fieldnames=["Loss", "Full_Loss", "tdist", "fid",
                                           "grad_norm", "overlap"])
    writer.writeheader()
    print(f'Created {filename}...')

    csv_writer = csv.writer(f, delimiter=',')

   # Adam paramenters
    alpha = lr
    beta_1 = 0.9
    beta_2 = 0.999      #initialize the values of the parameters
    epsilon = 1e-8      #initialize the vector
    m_t = 0 
    v_t = 0 
    for t in range(epochs):
        t+=1
        #print statements
        print('Overlap: ', over.real)
        print('Loss: ', loss.real)
        print('Full Loss: ', loss)
        print('Trace Distance: ', tdist)
        print('Fidelity: ', fid)

        #computes the gradients of OVERLAP
        g_t = model.backward(l2_reg_penalty=l2_reg)
        grad_norm =  np.abs(g_t).max()
        #update the moving average of the gradient
        m_t = beta_1*m_t + (1-beta_1)*g_t
        #updates the moving averages of the squared gradient
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t) 
        m_cap = m_t/(1-(beta_1**t))     #calculates the bias-corrected estimates
        v_cap = v_t/(1-(beta_2**t))     #calculates the bias-corrected estimates
        model.weights = model.weights - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)
##        Gradient Decesent
#        model.weights = model.weights - lr*g_t

        #gradient descent
        #model.weights = model.weights - lr*gradients
        sigma_v = model.vis_forward()
        loss = QRenyiDiv(sigma_v, rho)
##   Trace distance as a measure of test accuracy
        tdist = tracedist(sigma_v, rho)
        fid = fidelity(sigma_v, rho)
        over = rho.overlap(sigma_v)
        csv_writer.writerow([loss.real, loss, tdist, fid, grad_norm, over.real])
    f.close()

##    Unit Tests
#
#    1. Gradient is zero for identical states
#
#    rho = thermal_dm(2,2,'analytic')
#    rho = rho / rho.tr()
#    model = trotterized_model(2, 1, rho)
#    rho = model.forward().ptrace(model.vis_registers)
#    seed(42)
#    model = trotterized_model(4, 1, rho)
#
#
#    2. Verify numerical gradient is identical to theoretical gradient
#
#    seed(42)
#    rho = thermal_dm(2,2,'analytic')
#    rho = rho / rho.tr()
#    model = trotterized_model(2, 1, rho)
#    sigma_v = model.vis_forward()
#    theory_grad = [model.grad(i).real for i in range(model.num_params)]
#    org_weights = model.weights
#    eps = 0.0001
#    #estimated gradients
#    est_grad=[]
#    for i in range(model.num_params):
#        h = np.zeros(model.num_params)
#        h[i] = eps
#        update = (np.array(model.weights) + h).tolist()
#        model.weights = update
#        g = (QRenyiDiv(model.vis_forward(), rho) - QRenyiDiv(sigma_v, rho))/eps
#        est_grad.append(g)
#        model.weights = org_weights
#
#    3. Network is able to learn diagonal state.
#    seed(42)
#    rho = thermal_dm(2,2,'analytic')
#    rho = rho / rho.tr()
#    model = trotterized_model(2, 1, rho)
#
#    for _ in range(100):
#        sigma_v = model.vis_forward()
#        loss = QRenyiDiv(sigma_v, rho)
#        print('Loss: ', loss)
#        update = model.backward(lr=0.01)
#        model.weights = update
#
#    4. Network is able to learn random density.
#    seed(42)
#    num_registers = 2
#    vis_registers = 1
#    rho = rand_dm(vis_registers**2, density=0.75, dims=[[2,2] for _ in
#                                                        range(vis_registers)])
#    model = trotterized_model(num_registers, vis_registers, rho)
#    for _ in range(100):
#        sigma_v = model.vis_forward()
#        loss = QRenyiDiv(sigma_v, rho)
#        print('Loss: ', [l.real if abs(l.imag) < 1e-15 else l for l 
#                         in [loss]][0])
#        update = model.backward(lr=0.01)
#        model.weights = update
#
###   Trace distance as a measure of test accuracy
#    tdist = tracedist(sigma_v, rho)
#    print('Trace Distance between model and state: ', tdist)
