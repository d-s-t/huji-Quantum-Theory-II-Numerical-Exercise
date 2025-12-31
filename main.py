from eigen_state_solvers import finite_difference_method_radial, numerov_method_radial
from utils import plotly_export
from sys import float_info
import numpy as np
from plotly import graph_objects as go
from scipy.optimize import curve_fit

def error_func(N, C, q):
    return C * N**(-q)

def harmonic_oscillator_task(table_filename: str = 'harmonic_oscillator_results.tex'):
    """
    use both finite difference and numerov methods to solve the harmonic oscillator potential
    use l = 0, R = 10 and N= [40,80,120,240,360,480,600]
    1. plot the wave function of the ground state
    2. write latex table of the ground state energy epsilon for each N and each method.
       plot the residual error eta = |epsilon - 3/2| as a function of N. use log-log scale.
    3. curve fit the error to a function of the form eta = C * N^{-q} and report the values of C and q.
    """
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
        plotly_export(wave_function_fig, f'harmonic_oscillator\\wavefunction\\N{N}')
        
        ground_state_energies_fd[i] = evals_fd[0]
        ground_state_energies_nm[i] = evals_nm[0]
    
    # make one figure of only the ground state wave function for N=600 with numerov method
    wave_function_fig = go.Figure([go.Scatter(x=r, y=evecs_nm[:,0], mode='lines', name='Numerov')])\
                        .update_xaxes(title_text='r')\
                        .update_yaxes(title_text='u(r)')
    plotly_export(wave_function_fig, f'harmonic_oscillator\\ground_state_wavefunction_numerov_N600')

    # write latex table of the ground state energy \epsilon for each N and each method.
    with open(table_filename, 'w') as f:
        f.write(r'\begin{tabular}{|c|c|c|}' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'K & הפרשים סופיים & שיטת נומרוב \\' + '\n')
        f.write(r'\hline' + '\n')
        for N, e_fd, e_nm in zip(N_values, ground_state_energies_fd, ground_state_energies_nm):
            f.write(f'{N} & {e_fd:.6f} & {e_nm:.6f} \\\\' + '\n')
            f.write(r'\hline' + '\n')
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
    
    plotly_export(residual_fig, 'harmonic_oscillator\\residual_error')

    (_, q_fd), _ = curve_fit(error_func, N_values, residual_error_fd)
    (_, q_nm), _ = curve_fit(error_func, N_values, residual_error_nm)
    print(f'Finite Difference Method: q = {q_fd:.4f}')
    print(f'Numerov Method: q = {q_nm:.4f}')

def hydrogen_atom_task():
    N_values = [80, 120, 240, 360, 480, 600]
    l_values = np.array([0, 1, 2])
    R = 50
    from potential_functions import hydrogen_atom_potential as V
    ground_state_energies_fd = np.empty((len(l_values), len(N_values)), dtype=float)
    ground_state_energies_nm = np.empty((len(l_values), len(N_values)), dtype=float)
    for i, l in enumerate(l_values):
        for N in N_values:
            r = np.linspace(0, R, N + 1)[1:]
            evals_fd = finite_difference_method_radial(l, V, r, eigvals_only=True)
            evals_nm = numerov_method_radial(l, V, r, eigvals_only=True)
            ground_state_energies_fd[i, N_values.index(N)] = evals_fd[0]
            ground_state_energies_nm[i, N_values.index(N)] = evals_nm[0]
    
    with open('hydrogen_atom_results.tex', 'w') as f:
        f.write(r'\begin{tabular}{|c|' + 'c|'*len(N_values) + '}' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'l \textbackslash K ' + ' & ' + ' & '.join(str(N) for N in N_values) + r' \\' + '\n')
        f.write(r'\hline' + '\n')
        for i, l in enumerate(l_values):
            f.write(f'{l} (FD) ' + ' & ' + ' & '.join(f'{ground_state_energies_fd[i,j]:.6f}' for j in range(len(N_values))) + r' \\' + '\n')
            f.write(r'\hline' + '\n')
            f.write(f'{l} (NM) ' + ' & ' + ' & '.join(f'{ground_state_energies_nm[i,j]:.6f}' for j in range(len(N_values))) + r' \\' + '\n')
            f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')

    q_values_fd = []
    q_values_nm = []
    residual_error_fd = np.abs(ground_state_energies_fd + 0.5 / (l_values[:,None] + 1)**2)
    residual_error_nm = np.abs(ground_state_energies_nm + 0.5 / (l_values[:,None] + 1)**2)
    for i, l in enumerate(l_values):
        (_, q_fd), _ = curve_fit(error_func, N_values, residual_error_fd[i])
        (_, q_nm), _ = curve_fit(error_func, N_values, residual_error_nm[i])
        q_values_fd.append(q_fd)
        q_values_nm.append(q_nm)
    
    # write q values to latex table
    with open('hydrogen_atom_convergence_rates.tex', 'w') as f:
        f.write(r'\begin{tabular}{|c|c|c|}' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'l & q (FD) & q (NM) \\' + '\n')
        f.write(r'\hline' + '\n')
        for i, l in enumerate(l_values):
            f.write(f'{l} & {q_values_fd[i]:.4f} & {q_values_nm[i]:.4f} \\' + '\n')
            f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')




if __name__ == '__main__':
    # harmonic_oscillator_task()
    hydrogen_atom_task()