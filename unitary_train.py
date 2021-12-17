"""
Title: Training of a Unitary Quantum Network using the Renyi Divergence
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
from qutip import Qobj
from qutip.random_objects import rand_unitary_haar, rand_herm, rand_dm
from qutip.operators import sigmax, sigmay, sigmaz, identity
from qutip.tensor import tensor
from qutip.states import maximally_mixed_dm, ket2dm, basis, thermal_dm
from qutip.metrics import tracedist, fidelity

import tensorflow as tf

# Custom Modules
from target_states import two_body_ts

#tf-multi
def multidot_tf(matrices):
    mat_tf = tf.stack([tf.convert_to_tensor(p.full()) for p in matrices])
    return tf.scan(lambda a, b: tf.matmul(a, b), mat_tf)[-1].numpy()

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

class trotterized_model:
        def __init__(self, size, vis_size, rho, units_2x=False, no_y=False):
            self.size = size
            self.rho = rho
            self.vis_size = vis_size
            self.vis_registers = list(range(vis_size))
            self.rho_inv = rho.inv()
            self.units_2x = units_2x

            self.vac = tensor([basis(2, 0) for _ in range(size)])

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


            #randomly initialized
            self.weights = randn(self.num_params)


        def forward(self):
            prod_terms = [(complex(0,-1)*(w*p)).expm() for w,p in 
                          zip(self.weights, self.pauli_ops)]
            den_vec = self.vac
            sigma_vec=Qobj(multi_dot([*prod_terms, den_vec]))
            sigma_pred = ket2dm(sigma_vec)
            sigma_pred.dims = prod_terms[0].dims
            return sigma_pred

        def vis_forward(self):
            if self.vis_size == self.size:
                sigma_v = self.forward()
            else:
                sigma_v = self.forward().ptrace(self.vis_registers)
            return sigma_v

        def grad_tf(self, k, sigma, sigma_v, prod_terms_ineg, prod_terms_ipos):
            """not normalized"""
            #H^{\tilde}_k
            if k == 0:
                H_k_tilde = self.pauli_ops[k]
            else:
                H_k = self.pauli_ops[k]
                tilde_terms = prod_terms_ineg[:k]+[H_k]+\
                    list(reversed(prod_terms_ipos[:k]))
                H_k_tilde = Qobj(multidot_tf(tilde_terms), dims=H_k.dims)
            if self.vis_size == self.size:
                comm_ptrace = (H_k_tilde*sigma-\
                               sigma*H_k_tilde)
            else: 
                comm_ptrace = (H_k_tilde*sigma-\
                               sigma*H_k_tilde).ptrace(self.vis_registers)
            anti_comm = comm_ptrace*sigma_v + sigma_v*comm_ptrace
            mag_grad_k = complex(0,-1)*((anti_comm*self.rho_inv).tr())
            return mag_grad_k

        def grad(self, k, sigma, sigma_v, prod_terms_ineg, prod_terms_ipos):
            """not normalized"""
            #H^{\tilde}_k
            if k == 0:
                H_k_tilde = self.pauli_ops[k]
            else:
                H_k = self.pauli_ops[k]
                tilde_terms = prod_terms_ineg[:k]+[H_k]+\
                    list(reversed(prod_terms_ipos[:k]))
                H_k_tilde = Qobj(multi_dot(tilde_terms), dims=H_k.dims)
            if self.vis_size == self.size:
                comm_ptrace = (H_k_tilde*sigma-\
                               sigma*H_k_tilde)
            else: 
                comm_ptrace = (H_k_tilde*sigma-\
                               sigma*H_k_tilde).ptrace(self.vis_registers)
            anti_comm = comm_ptrace*sigma_v + sigma_v*comm_ptrace
            mag_grad_k = complex(0,-1)*((anti_comm*self.rho_inv).tr())
            return mag_grad_k

        def backward(self):
            """Returns the new weights."""
            prod_terms = [((w*p)) for w,p in 
                          zip(self.weights, self.pauli_ops)]
            prod_terms_neg = [(complex(0,-1)*term).expm() for term in 
                              prod_terms]
            prod_terms_pos = [(complex(0,1)*term).expm() for term in 
                                       prod_terms]
            sigma = self.forward()
            if self.vis_size == self.size:
                sigma_v = sigma
            else:
                sigma_v = sigma.ptrace(self.vis_registers)
            scale_of_grad = ((sigma_v*sigma_v)*self.rho_inv).tr()
            grads = np.array([self.grad_tf(k, sigma, sigma_v, prod_terms_neg, 
                                        prod_terms_pos) for k 
                              in range(len(self.weights))])
            print('Grad Update Complete...\n')
            if np.any(abs(grads.imag) > 1e-7):
                print("Warning: Gradients contain a non-zero complex part ",
                      "> 1e-7: ")


            return (grads.real)/(scale_of_grad.real)
            #check if gradients are real.
#            if not np.all(abs(updates.imag) < 1e-14):
#                print("Warning: Gradients contain a non-zero complex part:")
#                print(updates.imag)
#            return (updates.real.tolist(), 


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
    parser.add_argument('--s_arc', type=int, default=4, 
                        help='Random seed for neural architecture.')
    parser.add_argument('--s_state', type=int, default=7, 
                        help='Random seed for rho (training data).')
    parser.add_argument('--s_init', type=int, default=11, 
                        help='Random seed for network initialization.')
    parser.add_argument('--folder', type=str, default='./', 
                        help='Folder to store loss and accuracy.')


    args = parser.parse_args()
    vis_qb = args.vis_qb
    num_qb = args.num_qb
    units_2x = args.units_2x
    s_state = args.s_state
    s_arc = args.s_arc
    s_init = args.s_init
    lr=args.lr
    filename = args.folder+f'summary_num_{num_qb}_vis_{vis_qb}_2x_{units_2x}_'+\
    f'ss_{s_state}_sa_{s_arc}_si_{s_init}_lr_{lr}.csv'

    nps(s_state) #fix state
    rho = two_body_ts(0.1, size=vis_qb, normalize=True)
    model = trotterized_model(num_qb, vis_qb, rho, units_2x=units_2x)

    print('Number of model parameters...', model.num_params)

    #set initial weights
    nps(s_init)
    model.weights = randn(model.num_params)

    #shuffle hamiltonians terms
    rs(s_arc)
    model.pauli_ops = sample(model.pauli_ops, k=model.num_params)
    model.pauli_ops = sample(model.pauli_ops, k=model.num_params)
    model.pauli_ops = sample(model.pauli_ops, k=model.num_params)
    model.pauli_ops = sample(model.pauli_ops, k=model.num_params)

    sigma_v = model.vis_forward()

    loss = QRenyiDiv(sigma_v, rho)
    tdist = tracedist(sigma_v, rho)
    fid = fidelity(sigma_v, rho)


    f = open(filename, "a")
    writer = csv.DictWriter(f, fieldnames=["Loss", "Full_Loss", "tdist", "fid"])
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
    for t in range(2500):
        t+=1
        #print statements
        print('Loss: ', loss.real)
        print('Full Loss: ', loss)
        print('Trace Distance: ', tdist)
        print('Fidelity: ', fid)
        #computes the gradients
        g_t = model.backward()
        #update the moving average of the gradient
        m_t = beta_1*m_t + (1-beta_1)*g_t
        #updates the moving averages of the squared gradient
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t) 
        m_cap = m_t/(1-(beta_1**t))     #calculates the bias-corrected estimates
        v_cap = v_t/(1-(beta_2**t))     #calculates the bias-corrected estimates
        model.weights = model.weights - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)

        #gradient descent
        #model.weights = model.weights - lr*gradients
        sigma_v = model.vis_forward()
        loss = QRenyiDiv(sigma_v, rho)
##   Trace distance as a measure of test accuracy
        tdist = tracedist(sigma_v, rho)
        fid = fidelity(sigma_v, rho)
        csv_writer.writerow([loss.real, loss, tdist, fid])

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
