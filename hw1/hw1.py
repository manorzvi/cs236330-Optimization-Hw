import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def magic(n):
    '''
    A function to generate odd sized magic squares
    '''
    # 2-D array with all slots set to 0
    magicSquare = np.zeros((n,n))
    # initialize position of 1
    i = n / 2
    j = n - 1
    # Fill the magic square by placing values
    num = 1
    while num <= (n * n):
        if i == -1 and j == n:  # 3rd condition
            j = n - 2
            i = 0
        else:
            # next number goes out of right side of square
            if j == n: j = 0
            # next number goes out of upper side
            if i < 0: i = n - 1
        if magicSquare[int(i)][int(j)]:  # 2nd condition
            j = j - 2
            i = i + 1
            continue
        else:
            magicSquare[int(i)][int(j)] = num
            num = num + 1
        j = j + 1
        i = i - 1  # 1st condition
    return magicSquare

def phi(x):
    return np.sin(np.prod(x,axis=0))

def gradient_phi(x):
    ret_val = np.array([x[1]*x[2],
                        x[0]*x[2],
                        x[0]*x[1]]) * np.cos(np.prod(x,axis=0))
    return ret_val

def hessian_phi(x):
    cos_coef = np.array([[0,x[2],x[1]],
                         [x[2],0,x[0]],
                         [x[1],x[0],0]])
    cos = np.cos(np.prod(x,axis=0))
    # print('cos_coef\n',cos_coef,'\ncos_coef.shape\n',cos_coef.shape,'cos',cos)
    sin_coef = np.matmul(np.array([x[1]*x[2],
                                   x[0]*x[2],
                                   x[0]*x[1]]),
                         np.transpose(np.array([x[1]*x[2],
                                                x[0]*x[2],
                                                x[0]*x[1]])))
    sin = np.sin(np.prod(x,axis=0))
    # print('sin_coef\n',sin_coef,'\nsin_coef.shape\n',sin_coef.shape,'sin',sin)

    return cos_coef * cos - sin_coef * sin

def first_derivative_h(x):
    return np.exp(x)

def second_derivative_h(x):
    return np.exp(x)

class f1_par:
    def __init__(self, A, phi, gradient_phi, hessian_phi):
        self.A            = A
        self.gradient_phi = gradient_phi
        self.hessian_phi  = hessian_phi
        self.phi          = phi

def f1(x, par):
    '''
    f1(x) = phi(Ax) = sin(Ax[0]*Ax[1]*Ax[2])
    '''
    A        = par.A
    A_hat    = np.transpose(A)
    Ax       = np.matmul(A, x)
    # print('A',A)
    # print('x',x)
    # print('Ax',Ax)
    phi = par.phi(Ax)
    grad_phi = par.gradient_phi(Ax)
    hess_phi = par.hessian_phi(Ax)


    f   = phi
    # print('f',f)
    g   = np.matmul(A_hat,grad_phi)
    H   = np.matmul(np.matmul(A_hat,hess_phi),A)

    return f,g,H

class f2_par:
    def __init__(self, phi, gradient_phi, hessian_phi, first_derivative_h, second_derivative_h):
        self.gradient_phi        = gradient_phi
        self.hessian_phi         = hessian_phi
        self.phi                 = phi
        self.first_derivative_h  = first_derivative_h
        self.second_derivative_h = second_derivative_h

def f2(x, par):
    '''
    f2(x) = h(phi(x)) = exp(sin(x[0]*x[1]*x[2]))
    '''
    phi                 = par.phi(x)
    grad_phi            = par.gradient_phi(x)
    grad_phi_hat        = np.transpose(grad_phi)
    first_derivative_h  = par.first_derivative_h(phi)
    second_derivative_h = par.second_derivative_h(phi)
    hess_phi            = par.hessian_phi(x)

    h                   = np.exp(phi)
    g                   = first_derivative_h*grad_phi
    H                   = second_derivative_h*np.matmul(grad_phi, grad_phi_hat) + first_derivative_h*hess_phi
    return h,g,H

def run_f1(x, A):
    par = f1_par(A,phi,gradient_phi,hessian_phi)
    f, g, H_ = f1(x,par)
    H = np.zeros((H_.shape[0],H_.shape[0]))
    for i in range(H_.shape[0]):
        for j in range(H_.shape[1]):
            H[i,j] = H_[i][j]
    return f, g, H

def run_f2(x):
    par = f2_par(phi,gradient_phi,hessian_phi,first_derivative_h,second_derivative_h)
    h,g,H_ = f2(x,par)
    H = np.zeros((H_.shape[0], H_.shape[0]))
    for i in range(H_.shape[0]):
        for j in range(H_.shape[1]):
            H[i, j] = H_[i][j]

    return h, g, H

class numdiff_par:
    def __init__(self, x, phi, gradient_phi, hessian_phi, A=None, first_derivative_h=None, second_derivative_h=None,
                 epsilon_machine=2*(10**-16)):
        self.epsilon_machine = epsilon_machine
        self.eps_ = (epsilon_machine ** (1.0 / 3.0)) * max(np.abs(x))
        self.e   = np.identity(x.shape[0])
        self.A = A
        self.gradient_phi = gradient_phi
        self.hessian_phi = hessian_phi
        self.phi = phi
        self.first_derivative_h  = first_derivative_h
        self.second_derivative_h = second_derivative_h

def numdiff(myfunc, x, par):
    eps_ = par.eps_
    e   = par.e
    eps = eps_ * e
    dupx = np.repeat(x, x.shape[0], axis=1)
    dx_right  = dupx+eps
    dx_left = dupx-eps

    df_right = np.apply_along_axis(myfunc, 0, dx_right, par)[0,:]
    df_left  = np.apply_along_axis(myfunc, 0, dx_left, par)[0,:]

    gnum = (df_right-df_left)/(2*eps_)

    gf_right = np.apply_along_axis(myfunc, 0, dx_right, par)[1,:]
    gf_left  = np.apply_along_axis(myfunc, 0, dx_left, par)[1,:]

    Hnum = (gf_right-gf_left)/(2*eps_)

    return gnum, Hnum


def run_plots(value, gradient, hessian, gradient_numerical, hessian_numerical, mode, x):
    gradient_diff = np.abs(gradient-gradient_numerical)
    hessian_diff  = np.abs(hessian-hessian_numerical)

    fig,ax = plt.subplots(2,2)
    fig.suptitle(mode,fontsize=20)
    fig.tight_layout()

    ax[0,0].plot(gradient_diff, 'x')
    ax[0,0].set_title('Gradient Difference')
    ax[0,0].set_xlabel('Gradient elements indices')
    ax[0,0].set_ylabel('Absulute difference')
    ax[0,0].grid(True)
    im01 = ax[0,1].imshow(hessian_diff)
    ax[0,1].set_title('Hessian Difference')
    ax[0,1].set_xlabel('Hessian elements indices')
    ax[0,1].set_ylabel('Hessian elements indices')
    divider = make_axes_locatable(ax[0,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im01, cax=cax, orientation='vertical')


    eps_vec = np.logspace(-10.0, 0.0, num=100)
    # eps_vec = np.linspace(0, 0.1, num=100)

    func_dict = {'f1':f1,
                 'f2':f2}

    infinity_norm_g_diffs = []
    infinity_norm_H_diffs = []
    for eps_ in eps_vec:
        if mode == 'f1':
            par_ = numdiff_par(x=x, A=A, phi=phi, gradient_phi=gradient_phi, hessian_phi=hessian_phi)
        else:
            par_ = numdiff_par(x=x, phi=phi, gradient_phi=gradient_phi, hessian_phi=hessian_phi,
                               first_derivative_h=first_derivative_h, second_derivative_h=second_derivative_h)
        par_.eps_ = eps_
        gnum_, Hnum_ = run_numdiff(func_dict[mode], x, par_)
        infinity_norm_g_diffs.append(np.max(np.abs(gradient - gnum_)))
        infinity_norm_H_diffs.append(np.max(np.abs(hessian - Hnum_)))

    g_min_err = np.min(infinity_norm_g_diffs)
    g_min_err_ind = np.argmin(infinity_norm_g_diffs)
    H_min_err = np.min(infinity_norm_H_diffs)
    H_min_err_ind = np.argmin(infinity_norm_H_diffs)

    ax[1, 0].plot(eps_vec, infinity_norm_g_diffs, 'x')
    ax[1, 0].set_title('Inf Norm Gradient Difference')
    ax[1, 0].set_xlabel('Epsilon (Log scale)')
    ax[1, 0].set_ylabel('Inf. Norm of the Absulute difference')
    ax[1, 0].grid(True)
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_yscale('log')
    ax[1, 0].plot(eps_vec[g_min_err_ind], g_min_err, "*", color='red')
    min_err_string = "{:.5f}".format(g_min_err)
    ax[1,0].annotate(f'min value {min_err_string}', (eps_vec[g_min_err_ind], g_min_err))

    ax[1, 1].plot(eps_vec, infinity_norm_H_diffs, 'x')
    ax[1, 1].set_title('Inf Norm Hessian Difference')
    ax[1, 1].set_xlabel('Epsilon (Log Scale)')
    ax[1, 1].set_ylabel('Inf. Norm of the Absulute difference')
    ax[1, 1].grid(True)
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_yscale('log')
    ax[1, 1].plot(eps_vec[H_min_err_ind], H_min_err, "*", color='red')
    min_err_string = "{:.5f}".format(H_min_err)
    ax[1, 1].annotate(f'min value {min_err_string}', (eps_vec[H_min_err_ind], H_min_err))
    plt.show()


def run_numdiff(myfunc, x, par):
    gnum_, Hnum_ = numdiff(myfunc, x, par)
    Hnum = np.zeros((Hnum_.shape[0],Hnum_.shape[0]))
    gnum = np.zeros((gnum_.shape[0],1))
    for i in range(Hh.shape[0]):
        for j in range(Hh.shape[1]):
            Hnum[i,j] = Hnum_[i][j]
        gnum[i] = gnum_[i]
    return gnum, Hnum

if __name__ == '__main__':
    n = 3
    A = magic(n)
    x = np.array([[1],[2],[3]])
    # A = np.identity(x.shape[0])
    print(f'Magic Matrix:\n{A}\n\n')
    print(f'Input Vector:\n{x}\n Of Size: {x.shape}\n\n')
    print(f'\nTask 3 part a\n'
          f'-------------\n')
    f, gf, Hf = run_f1(x,A)
    print('\nf\n',f,'\ng\n',gf,'\nH\n',Hf)
    print(f'\nTask 3 part b\n'
          f'-------------\n')
    h, gh, Hh = run_f2(x)
    print('\nf\n',h,'\ng\n',gh,'\nH\n',Hh)
    print('\n\nRun Task 4 with f1:\n'
          '-------------------')
    par1 = numdiff_par(x=x, A=A, phi=phi, gradient_phi=gradient_phi, hessian_phi=hessian_phi)
    gnum1, Hnum1 = run_numdiff(f1, x, par1)
    print('\nNumerical Gradient:\n', gnum1)
    print('\nNumerical Hessian:\n', Hnum1)
    print('\n\nRun Task 4 with f2:\n'
          '-------------------')
    par2 = numdiff_par(x=x, phi=phi, gradient_phi=gradient_phi, hessian_phi=hessian_phi,
                       first_derivative_h=first_derivative_h, second_derivative_h=second_derivative_h)
    gnum2, Hnum2 = run_numdiff(f2, x, par2)
    print('\nNumerical Gradient:\n', gnum2)
    print('\nNumerical Hessian:\n', Hnum2)

    print('\n\nRun Task 5 with f1:\n'
          '-------------------')
    run_plots(f, gf, Hf, gnum1, Hnum1, 'f1', x)

    print('\n\nRun Task 5 with f2:\n'
          '-------------------')
    run_plots(h, gh, Hh, gnum2, Hnum2, 'f2', x)