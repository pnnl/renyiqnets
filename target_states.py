"""
Target State Module

Class of thermal states used for our learning tasks.

"""

from itertools import product
from numpy.linalg import norm
from numpy.random import normal as N
from numpy.random import randn
from numpy.random import seed
from qutip.operators import sigmax, sigmay, sigmaz, identity
from qutip.tensor import tensor


def tensor_sigmas(positions, paulis, size):
    Is=[identity(2) for _ in range(size-len(positions))]
    for p, pauli in zip(positions, paulis):
        Is.insert(p, pauli)
    return tensor(Is)

def two_body_ts(sigma_onsite, size, normalize, temp=1):
    loop_z=sum([N(scale=sigma_onsite)*tensor_sigmas([i], [sigmaz()], size=size) for i in 
              range(size)])
    loop_y=sum([N(scale=sigma_onsite)*tensor_sigmas([i], [sigmay()], size=size) for i in 
              range(size)])
    loop_x=sum([N(scale=sigma_onsite)*tensor_sigmas([i], [sigmax()], size=size) for i in 
              range(size)])

    offdiag=[(i,j) for i, j in list(product(range(size), range(size))) if i<j]

    inter_x=sum([N()*tensor_sigmas([i,j], [sigmax(),sigmax()],size=size) for
                 i, j in offdiag])
    inter_y=sum([N()*tensor_sigmas([i,j], [sigmay(),sigmay()],size=size) for
                 i, j in offdiag])
    inter_z=sum([N()*tensor_sigmas([i,j], [sigmaz(),sigmaz()],size=size) for
                 i, j in offdiag])
    inter_xy=sum([N()*tensor_sigmas([i,j], [sigmax(),sigmay()],size=size) for
                 i, j in offdiag])
    inter_xz=sum([N()*tensor_sigmas([i,j], [sigmax(),sigmaz()],size=size) for
                 i, j in offdiag])
    inter_yz=sum([N()*tensor_sigmas([i,j], [sigmax(),sigmaz()],size=size) for
                 i, j in offdiag])
    inter_yx=sum([N()*tensor_sigmas([i,j], [sigmay(),sigmax()],size=size) for
                 i, j in offdiag])
    inter_zx=sum([N()*tensor_sigmas([i,j], [sigmaz(),sigmax()],size=size) for
                 i, j in offdiag])
    inter_zy=sum([N()*tensor_sigmas([i,j], [sigmaz(),sigmay()],size=size) for
                 i, j in offdiag])
    H = -1*(loop_x+loop_y+loop_z+inter_x+inter_y+inter_z+inter_xy+\
                       inter_xz+inter_yz+inter_yx+inter_zx+inter_zy)

    if normalize:
        print('Normalized...', normalize)
        H_opnorm = norm(H, ord=2)
        H = temp*(H/H_opnorm)

    expH = H.expm()
    norm_expH = expH / expH.tr()
    return norm_expH

def three_body_ts(size, temp=5):

     loop_z=[tensor_sigmas([i], [sigmaz()], size=size) 
             for i in range(size)]
     loop_y=[tensor_sigmas([i], [sigmay()], size=size) 
             for i in range(size)]
     loop_x=[tensor_sigmas([i], [sigmax()], size=size) 
             for i in range(size)]

     offdiag=[(i,j) for i, j in list(product(range(size), range(size))) 
              if i<j]
     offdiag_3=[(i,j,k) for i, j, k in 
              list(product(range(size), range(size), range(size))) 
              if (i<j and j<k)]

     inter_xx=[tensor_sigmas([i,j], [sigmax(),sigmax()],size=size) for
              i, j in offdiag]
     inter_yy=[tensor_sigmas([i,j], [sigmay(),sigmay()],size=size) for
              i, j in offdiag]
     inter_zz=[tensor_sigmas([i,j], [sigmaz(),sigmaz()],size=size) for
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
     inter_xxx = [tensor_sigmas([i,j,k], [sigmax(),sigmax(), sigmax()], 
                                size=size) for i, j, k in offdiag_3]
     inter_yyy = [tensor_sigmas([i,j,k], [sigmay(),sigmay(), sigmay()], 
                                size=size) for i, j, k in offdiag_3]
     inter_zzz = [tensor_sigmas([i,j,k], [sigmaz(),sigmaz(), sigmaz()], 
                                size=size) for i, j, k in offdiag_3]
     inter_yxx = [tensor_sigmas([i,j,k], [sigmay(),sigmax(), sigmax()], 
                                size=size) for i, j, k in offdiag_3]
     inter_xyx = [tensor_sigmas([i,j,k], [sigmax(),sigmay(), sigmax()], 
                                size=size) for i, j, k in offdiag_3]
     inter_xxy = [tensor_sigmas([i,j,k], [sigmax(),sigmax(), sigmay()], 
                                size=size) for i, j, k in offdiag_3]
     inter_zxx = [tensor_sigmas([i,j,k], [sigmaz(),sigmax(), sigmax()], 
                                size=size) for i, j, k in offdiag_3]
     inter_xzx = [tensor_sigmas([i,j,k], [sigmax(),sigmaz(), sigmax()], 
                                size=size) for i, j, k in offdiag_3]
     inter_xxz = [tensor_sigmas([i,j,k], [sigmax(),sigmax(), sigmaz()], 
                                size=size) for i, j, k in offdiag_3]
     inter_yzz = [tensor_sigmas([i,j,k], [sigmay(),sigmaz(), sigmaz()], 
                                size=size) for i, j, k in offdiag_3]
     inter_zyz = [tensor_sigmas([i,j,k], [sigmaz(),sigmay(), sigmaz()], 
                                size=size) for i, j, k in offdiag_3]
     inter_zzy = [tensor_sigmas([i,j,k], [sigmaz(),sigmaz(), sigmay()], 
                                size=size) for i, j, k in offdiag_3]
     inter_zyy = [tensor_sigmas([i,j,k], [sigmaz(),sigmay(), sigmay()], 
                                size=size) for i, j, k in offdiag_3]
     inter_yzy = [tensor_sigmas([i,j,k], [sigmay(),sigmaz(), sigmay()], 
                                size=size) for i, j, k in offdiag_3]
     inter_yyz = [tensor_sigmas([i,j,k], [sigmay(),sigmay(), sigmaz()], 
                                size=size) for i, j, k in offdiag_3]
     inter_xyy = [tensor_sigmas([i,j,k], [sigmax(),sigmay(), sigmay()], 
                                size=size) for i, j, k in offdiag_3]
     inter_yxy = [tensor_sigmas([i,j,k], [sigmay(),sigmax(), sigmay()], 
                                size=size) for i, j, k in offdiag_3]
     inter_yyx = [tensor_sigmas([i,j,k], [sigmay(),sigmay(), sigmax()], 
                                size=size) for i, j, k in offdiag_3]
     inter_xzz = [tensor_sigmas([i,j,k], [sigmax(),sigmaz(), sigmaz()], 
                                size=size) for i, j, k in offdiag_3]
     inter_zxz = [tensor_sigmas([i,j,k], [sigmaz(),sigmax(), sigmaz()], 
                                size=size) for i, j, k in offdiag_3]
     inter_zzx = [tensor_sigmas([i,j,k], [sigmaz(),sigmaz(), sigmax()], 
                                size=size) for i, j, k in offdiag_3]
     inter_xyz = [tensor_sigmas([i,j,k], [sigmax(),sigmay(), sigmaz()], 
                                size=size) for i, j, k in offdiag_3]
     inter_xzy = [tensor_sigmas([i,j,k], [sigmax(),sigmaz(), sigmay()], 
                                size=size) for i, j, k in offdiag_3]
     inter_yxz = [tensor_sigmas([i,j,k], [sigmay(),sigmax(), sigmaz()], 
                                size=size) for i, j, k in offdiag_3]
     inter_yzx = [tensor_sigmas([i,j,k], [sigmay(),sigmaz(), sigmax()], 
                                size=size) for i, j, k in offdiag_3]
     inter_zxy = [tensor_sigmas([i,j,k], [sigmaz(),sigmax(), sigmay()], 
                                size=size) for i, j, k in offdiag_3]
     inter_zyx = [tensor_sigmas([i,j,k], [sigmaz(),sigmay(), sigmax()], 
                                size=size) for i, j, k in offdiag_3]

     pauli_ops = loop_x+loop_y+loop_z+inter_xx+inter_yy+\
                      inter_zz+inter_xy+inter_xz+inter_yz+\
                      inter_yx+inter_zx+inter_zy+inter_xxx+\
                      inter_yyy+inter_zzz+inter_yxx+inter_xyx+\
                      inter_xxy+inter_zxx+inter_xzx+inter_xxz+\
                      inter_yzz+inter_zyz+inter_zzy+inter_zyy+\
                      inter_yzy+inter_yyz+inter_xyy+inter_yxy+\
                      inter_yyx+inter_xzz+inter_zxz+inter_zzx+\
                      inter_xyz+inter_xzy+inter_yxz+inter_yzx+\
                      inter_zxy+inter_zyx

     num_params = len(pauli_ops)


     # randomly initialized, but normalized to prevent numerical
     # instabilities matrix inversion
     init_weights = randn(num_params)
     prod_terms = [(w*p) for w,p in 
                   zip(init_weights, pauli_ops)]
     H = -1*(sum(prod_terms))
     H_opnorm = norm(H, ord=2)
     weights = (temp/H_opnorm)*init_weights

     prod_terms = [(w*p) for w,p in 
                   zip(weights, pauli_ops)]
     expH = ((-1)*sum(prod_terms)).expm()
     norm_expH = expH / expH.tr()
     return norm_expH
