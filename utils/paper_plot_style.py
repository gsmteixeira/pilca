import matplotlib.pyplot as plt

def set_plot_style():
    plt.rcParams.update({
        # Font settings
        'font.size': 20,
        'legend.fontsize': 16,
        'font.family': 'serif',
        'font.weight': 'normal',
        'axes.labelweight': 'normal',
        'axes.labelsize': 22,

        # X-axis
        'xtick.direction': 'in',
        'xtick.major.size': 7,
        'xtick.minor.size': 4,
        'xtick.major.pad': 8,
        'xtick.minor.pad': 8,
        'xtick.major.width': 2,
        'xtick.minor.width': 2,
        'xtick.minor.visible': True,

        # Y-axis
        'ytick.direction': 'in',
        'ytick.major.size': 7,
        'ytick.minor.size': 4,
        'ytick.major.pad': 8,
        'ytick.minor.pad': 8,
        'ytick.major.width': 2,
        'ytick.minor.width': 2,
        'ytick.minor.visible': True,

        # general
        # 'legend.borderaxespad': 1.5,
        'axes.labelpad': 4,       # Default is 4; increase to push label away from axis

        })
    



def set_thesis_style(fontsize=20):

    plt.rcParams.update({

        # Font
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": fontsize,

        # Axes
        "axes.labelsize": fontsize * 1.2,
        "axes.titlesize": fontsize * 1.2,
        "axes.linewidth": 1.2,

        # Ticks
        "xtick.labelsize": fontsize * 0.9,
        "ytick.labelsize": fontsize * 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,

        # Legend
        "legend.fontsize": fontsize * 0.9,
        "legend.frameon": False,

        # Lines
        "lines.linewidth": 3,

        # Figure
        "figure.figsize": (6,4),
        "figure.dpi": 120,

        # Savefig
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # Grid
        "axes.grid": False
    })
    