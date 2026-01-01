from eigen_state_solvers import finite_difference_method_radial, numerov_method_radial, numerov_method_coulomb_l0_1st_order, numerov_method_coulomb_l0_2nd_order
from utils import plotly_export
import numpy as np
from plotly import graph_objects as go
from scipy.optimize import curve_fit
from os import path, makedirs

def error_func(N, C, q):
    return C * N**(-q)

def harmonic_oscillator_task(folder_name: str = 'harmonic_oscillator'):
    """
    use both finite difference and numerov methods to solve the harmonic oscillator potential
    use l = 0, R = 10 and N= [40,80,120,240,360,480,600]
    1. plot the wave function of the ground state
    2. write latex table of the ground state energy epsilon for each N and each method.
       plot the residual error eta = |epsilon - 3/2| as a function of N. use log-log scale.
    3. curve fit the error to a function of the form eta = C * N^{-q} and report the values of C and q.
    """
    makedirs(path.join('results', folder_name), exist_ok=True)

    l = 0
    R = 10
    N_values = [40,80,120,240,360,480,600]
    from potential_functions import harmonic_oscillator_potential as V
    ground_state_energies_fd = np.empty_like(N_values, dtype=float)
    ground_state_energies_nm = np.empty_like(N_values, dtype=float)
    for i, N in enumerate(N_values):
        r = np.linspace(0, R, N + 1)[1:]
        evals_fd, evecs_fd = finite_difference_method_radial(l, V, r)
        evals_nm, evecs_nm = numerov_method_radial(l, V, r)

        # plot the wave function of the ground state
        wave_function_fig = go.Figure([go.Scatter(x=r, y=evecs_fd[:,0], mode='lines', name='Finite Difference'),
                            go.Scatter(x=r, y=evecs_nm[:,0], mode='lines', name='Numerov')])\
                        .update_xaxes(title_text='r')\
                        .update_yaxes(title_text='u(r)')\
                        .update_layout(legend=dict(title='Method', yanchor="top", y=0.99, xanchor="right", x=0.99))
        plotly_export(wave_function_fig, path.join(folder_name, 'wavefunction', f'N{N}'))
        
        ground_state_energies_fd[i] = evals_fd[0]
        ground_state_energies_nm[i] = evals_nm[0]
    
    # make one figure of only the ground state wave function for N=600 with numerov method
    wave_function_fig = go.Figure([go.Scatter(x=r, y=evecs_nm[:,0], mode='lines', name='Numerov')])\
                        .update_xaxes(title_text='r')\
                        .update_yaxes(title_text='u(r)')
    plotly_export(wave_function_fig, path.join(folder_name, 'ground_state_wavefunction_numerov_N600'))

    # write latex table of the ground state energy \epsilon for each N and each method.
    with open(path.join('results', folder_name, 'ground_state_energies.tex'), 'w') as f:
        f.write(r'\begin{tabular}{c c c}' + '\n')
        # f.write(r'\hline' + '\n')
        f.write(r'K & הפרשים סופיים & שיטת נומרוב \\' + '\n')
        f.write(r'\hline' + '\n')
        for N, e_fd, e_nm in zip(N_values, ground_state_energies_fd, ground_state_energies_nm):
            f.write(f'{N} & {e_fd:.6g} & {e_nm:.6g} \\\\' + '\n')
            # f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')
    
    # plot the residual error \eta = |\epsilon - 3/2| as a function of N. use log-log scale.
    exact_energy = 1.5  # MeV
    residual_error_fd = np.abs(ground_state_energies_fd - exact_energy)
    residual_error_nm = np.abs(ground_state_energies_nm - exact_energy)

    residual_fig = go.Figure([go.Scatter(x=N_values, y=residual_error_fd, mode='markers+lines', name='Finite Difference'),
                   go.Scatter(x=N_values, y=residual_error_nm, mode='markers+lines', name='Numerov')])\
                .update_xaxes(title_text='N', type='log')\
                .update_yaxes(title_text='Residual Error (MeV)', type='log', showexponent='all', exponentformat='power')\
                .update_layout(legend=dict(title='Method'))
    
    plotly_export(residual_fig, path.join(folder_name, 'residual_error'))

    (_, q_fd), _ = curve_fit(error_func, N_values, residual_error_fd)
    (_, q_nm), _ = curve_fit(error_func, N_values, residual_error_nm)
    with open(path.join('results', folder_name, 'q_fd.tex'), 'w') as f:
        f.write(f'$q = {q_fd:.4f}$')
    with open(path.join('results', folder_name, 'q_nm.tex'), 'w') as f:
        f.write(f'$q = {q_nm:.4f}$')

def hydrogen_atom_task(folder_name: str = 'hydrogen_atom'):
    makedirs(path.join('results', folder_name), exist_ok=True)

    N_values = [80, 120, 240, 360, 480, 600]
    l_values = np.array([0, 1, 2])
    R = 50
    from potential_functions import hydrogen_atom_potential as V
    ground_state_energies_fd = np.empty((len(l_values), len(N_values)), dtype=float)
    ground_state_energies_nm = np.empty((len(l_values), len(N_values)), dtype=float)
    for i, l in enumerate(l_values):
        for j, N in enumerate(N_values):
            r = np.linspace(0, R, N + 1)[1:]
            evals_fd = finite_difference_method_radial(l, V, r, eigvals_only=True)
            evals_nm = numerov_method_radial(l, V, r, eigvals_only=True)
            ground_state_energies_fd[i, j] = evals_fd[0]
            ground_state_energies_nm[i, j] = evals_nm[0]
    
    with open(path.join('results', folder_name, 'ground_state_energies.tex'), 'w') as f:
        f.write(r'\begin{tabular}{|c|' + 'c|'*len(N_values) + '}' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'\diagbox[dir=NE, innerwidth = 3cm, height = 4ex]{$K$}{$l$}' + ' & ' + ' & '.join(str(N) for N in N_values) + r' \\' + '\n')
        f.write(r'\hline' + '\n')
        for i, l in enumerate(l_values):
            f.write(f'{l} (FD) ' + ' & ' + ' & '.join(f'${ground_state_energies_fd[i,j]:.4g}$' for j in range(len(N_values))) + r' \\' + '\n')
            # f.write(r'\hline' + '\n')
            f.write(f'\\ (NM) ' + ' & ' + ' & '.join(f'${ground_state_energies_nm[i,j]:.4g}$' for j in range(len(N_values))) + r' \\' + '\n')
            f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')

    residual_error_fd = np.abs(ground_state_energies_fd + 0.5 / (l_values[:,None] + 1)**2)
    residual_error_nm = np.abs(ground_state_energies_nm + 0.5 / (l_values[:,None] + 1)**2)

    residual_error_fig = go.Figure([go.Scatter(x=N_values, y=residual_error_fd[i], mode='markers+lines', name=f'FD l={l_values[i]}') for i in range(len(l_values))] +
                                   [go.Scatter(x=N_values, y=residual_error_nm[i], mode='markers+lines', name=f'NM l={l_values[i]}') for i in range(len(l_values))])\
                .update_xaxes(title_text='N', type='log')\
                .update_yaxes(title_text='Residual Error (a.u.)', type='log', showexponent='all', exponentformat='power')\
                .update_layout(legend=dict(title='Method and l'))
    
    plotly_export(residual_error_fig, path.join(folder_name, 'residual_error'))


    q_values_fd = []
    q_values_nm = []
    for i, l in enumerate(l_values):
        (_, q_fd), _ = curve_fit(error_func, N_values, residual_error_fd[i])
        (_, q_nm), _ = curve_fit(error_func, N_values, residual_error_nm[i])
        q_values_fd.append(q_fd)
        q_values_nm.append(q_nm)
    
    # write q values to latex table
    with open(path.join('results', folder_name, 'convergence_rates.tex'), 'w') as f:
        f.write(r'\begin{tabular}{c c c}' + '\n')
        # f.write(r'\hline' + '\n')
        f.write(r'$l$ & q (FD) & q (NM) \\' + '\n')
        f.write(r'\hline' + '\n')
        for i, l in enumerate(l_values):
            f.write(f'{l} & ${q_values_fd[i]:.6g}$ & ${q_values_nm[i]:.6g}$' + r' \\' + '\n')
            # f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')


def first_order_task(folder_name: str = 'first_order'):
    """
    do very simmilar to hydrogen atom task but use only numerov_method_coulomb_l0_1st_order(Z: int, r: np.ndarray, *, eigvals_only: bool = False)
    """
    makedirs(path.join('results', folder_name), exist_ok=True)

    N_values = [80, 120, 240, 360, 480, 600]
    R = 50
    Z = 1 # For hydrogen atom
    from potential_functions import hydrogen_atom_potential as V
    
    ground_state_energies_nm_l0_first_order = np.empty_like(N_values, dtype=float)
    ground_state_energies_nm_l0_regular = np.empty_like(N_values, dtype=float)
    
    for i, N in enumerate(N_values):
        r = np.linspace(0, R, N + 1)[1:]
        evals_nm = numerov_method_coulomb_l0_1st_order(Z, R, N, eigvals_only=True)
        ground_state_energies_nm_l0_first_order[i] = evals_nm[0]
        evals_nm_regular = numerov_method_radial(0, V, r, eigvals_only=True)
        ground_state_energies_nm_l0_regular[i] = evals_nm_regular[0]
            
    with open(path.join('results', folder_name, 'ground_state_energies.tex'), 'w') as f:
        f.write(r'\begin{tabular}{c c c}' + '\n')
        # f.write(r'\hline' + '\n')
        f.write(r'$K$ & שיטת נומרוב (סדר ראשון) & שיטת נומרוב \\' + '\n')
        f.write(r'\hline' + '\n')
        for i, N in enumerate(N_values):
            f.write(f'{N} & ${ground_state_energies_nm_l0_first_order[i]:.6g}$ & ${ground_state_energies_nm_l0_regular[i]:.6g}$ \\\\' + '\n')
            # f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')

    exact_energy_l0 = -0.5 # for l=0, n=1, E = -Z^2/(2n^2) = -1^2/(2*1^2) = -0.5 a.u.
    residual_error_nm_l0_first_order = np.abs(ground_state_energies_nm_l0_first_order - exact_energy_l0)
    residual_error_nm_l0_regular = np.abs(ground_state_energies_nm_l0_regular - exact_energy_l0)

    residual_error_fig = go.Figure([
        go.Scatter(x=N_values, y=residual_error_nm_l0_first_order, mode='markers+lines', name=f'NM (First Order)'),
        go.Scatter(x=N_values, y=residual_error_nm_l0_regular, mode='markers+lines', name=f'NM (Regular)')
    ])\
                .update_xaxes(title_text='N', type='log')\
                .update_yaxes(title_text='Residual Error (a.u.)', type='log', showexponent='all', exponentformat='power')\
                .update_layout(legend=dict(title='Method and l'))
    
    plotly_export(residual_error_fig, path.join(folder_name, 'residual_error'))

    (_, q_nm_o1), _ = curve_fit(error_func, N_values, residual_error_nm_l0_first_order)
    (_, q_nm), _ = curve_fit(error_func, N_values, residual_error_nm_l0_regular)
    with open(path.join('results' ,folder_name, 'q_nm_o1.tex'), 'w') as f:
        f.write(f'$q = {q_nm_o1:.4f}$')
    with open(path.join('results', folder_name, 'q_nm.tex'), 'w') as f:
        f.write(f'$q = {q_nm:.4f}$')

def second_order_task(folder_name: str = 'second_order'):
    """
    do the same as the first order task but use numerov_method_coulomb_l0_2nd_order
    compare the results to the first order method and to the regular numerov method
    """
    makedirs(path.join('results', folder_name), exist_ok=True)
    N_values = [80, 120, 240, 360, 480, 600]
    R = 50
    Z = 1 # For hydrogen atom
    from potential_functions import hydrogen_atom_potential as V
    
    ground_state_energies_nm_l0 = np.fromiter((numerov_method_radial(0, V, np.linspace(0, R, N + 1)[1:], eigvals_only=True)[0] for N in N_values), dtype=float)
    ground_state_energies_nm_l0_first_order = np.fromiter((numerov_method_coulomb_l0_1st_order(Z, R, N, eigvals_only=True)[0] for N in N_values), dtype=float)
    ground_state_energies_nm_l0_second_order = np.fromiter((numerov_method_coulomb_l0_2nd_order(Z, R, N, eigvals_only=True)[0] for N in N_values), dtype=float)

    exact_energy_l0 = -0.5 # for l=0, n=1, E = -Z^2/(2n^2) = -1^2/(2*1^2) = -0.5 a.u.
    residual_error_nm_l0 = np.abs(ground_state_energies_nm_l0 - exact_energy_l0)
    residual_error_nm_l0_first_order = np.abs(ground_state_energies_nm_l0_first_order - exact_energy_l0)
    residual_error_nm_l0_second_order = np.abs(ground_state_energies_nm_l0_second_order - exact_energy_l0)  
    residual_error_fig = go.Figure([
        go.Scatter(x=N_values, y=residual_error_nm_l0_second_order, mode='markers+lines', name=f'NM (Second Order)'),
        go.Scatter(x=N_values, y=residual_error_nm_l0_first_order, mode='markers+lines', name=f'NM (First Order)'),
        go.Scatter(x=N_values, y=residual_error_nm_l0, mode='markers+lines', name=f'NM (Regular)')
    ])\
                .update_xaxes(title_text='N', type='log')\
                .update_yaxes(title_text='Residual Error (a.u.)', type='log', showexponent='all', exponentformat='power')\
                .update_layout(legend=dict(title='Method and l'))
    plotly_export(residual_error_fig, path.join(folder_name, 'residual_error'))
    (_, q_nm), _ = curve_fit(error_func, N_values, residual_error_nm_l0)
    (_, q_nm_o1), _ = curve_fit(error_func, N_values, residual_error_nm_l0_first_order)
    (_, q_nm_o2), _ = curve_fit(error_func, N_values, residual_error_nm_l0_second_order)
    with open(path.join('results' ,folder_name, 'convergence_rates.tex'), 'w') as f:
        f.write(r'\begin{tabular}{c c}' + '\n')
        # f.write(r'\hline' + '\n')
        f.write(r'Method & $q$ \\' + '\n')
        f.write(f'שיטת נומרוב & ${q_nm:.6g}$ \\\\' + '\n')
        f.write(f'שיטת נומרוב (סדר ראשון) & ${q_nm_o1:.6g}$ \\\\' + '\n')
        f.write(f'שיטת נומרוב (סדר שני) & ${q_nm_o2:.6g}$ \\\\' + '\n')




if __name__ == '__main__':
    harmonic_oscillator_task()
    hydrogen_atom_task()
    first_order_task()
    second_order_task()