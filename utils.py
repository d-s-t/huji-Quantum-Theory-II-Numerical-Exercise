from astropy import units
from astropy.units.quantity import Quantity
import astropy.constants as const
import functools
import operator
import numpy as np
from os import path, makedirs

units.MeVc2 = units.def_unit('MeV / c2', units.MeV/const.c**2, format={'latex': r'MeV/c^2'})

const.nuclee_mass_estimation = 931.49432 * units.MeVc2

const.hbarc = (const.hbar * const.c).to(units.MeV * units.fm)

plotly_show_config = {
    'toImageButtonOptions': {
        'format': 'svg',
        'filename': 'unset',
        'width': 800, 'height': 450
    },
    "editable": True,
    # "addButtonToModeBar": [
    #     {
    #         'name': 'save_png',
    #         'title': 'Save as PNG',
    #         'icon': ,
    #         'click': pio.base_renderers.PngRenderer(800, 450, 2).to_mimebundle#lambda fig: pio.write_image(fig, f"./plots/{plotly_show_config['toImageButtonOptions']['filename']}.png", width=800, height=450, format='png', engine='kaleido', scale=2)
    #     }
    # ]
}

# make new template with transparent background
# pio.templates.add(go.layout.Template(name='transparent', layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)')))



def to_latex(q: Quantity, f:str):
    v = format(q.value, f)
    if 'e' in v:
        v,ex = v.split('e')
        v = fr"{v} \times 10^{{{ex}}}"
    return rf"{v} \; {q.unit._repr_latex_()[1:-1]}"

def flatten(a):
    return functools.reduce(operator.iconcat, a, [])


def plotly_export(fig, filename, html=False, show=False, **kwargs):
    dir = path.dirname(f'./plots/{filename}')
    if not path.exists(dir):
        makedirs(dir)
    fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), width=800, height=450)
    fig.write_image(f"./plots/{filename}.eps", width=500, height=300,format='eps', engine='kaleido')
    plotly_show_config['toImageButtonOptions']['filename'] = filename
    if html:
        fig.write_html(f"./plots/{filename}.html", config=plotly_show_config)
    if show:
        fig.show(config=plotly_show_config)
    plotly_show_config['toImageButtonOptions']['filename'] = 'unset'

def derivative_o4(u, dx):
    """
    Calculate the derivative of u with respect to x with acuracy O(dx^4)
    """
    return (u[:-4] - 8*u[1:-3] + 8*u[3:-1] - u[4:]) / (12*dx)

def progress_bar_range(*n):
    """
    this is generator that uses IPython.display to show a progress bar for the range
    """
    from tqdm import trange
    return trange(*n)

def relative_error(a: np.ndarray, b: np.ndarray,/) -> np.ndarray:
    """
    """
    return abs(1 - a / b)

def reduced_mass(*masses):
    """
    reduced mass of a system
    """
    return 1 / sum(1/m for m in masses)


def harmonic_oscillator_potential(r: np.ndarray) -> np.ndarray:
    """
    Example potential: Harmonic oscillator potential
    V(r) = r^2 / 2
    r: np.ndarray[float]
        Array of distances in fm
        shape: (N,)
    """
    return r**2/2


### <u|u> calculation
def inner_product(u1: np.ndarray, u2: np.ndarray, r: np.ndarray) -> float:
    """
    Calculate the inner product of two wave functions

    u1: np.ndarray[float]
        First wave function
        shape: (N,)
    u2: np.ndarray[float]
        Second wave function
        shape: (N,)
    r: np.ndarray[float]
        Array of distances in fm
        shape: (N,)

    returns: float
        Inner product <u1|u2>
    """
    return np.trapezoid(u1 * np.conjugate(u2), r)