import os
import matplotlib.pyplot as plt
import matplotlib as mpl


def set_plotstyle(width=7, height=15 * 3.35 / 16):
    os.environ['PATH'] = os.environ['PATH'] + "C:/Users/FischerVicto/AppData/Local/Programs/MiKTeX/miktex/bin/x64/"
    os.environ['PATH'] = os.environ['PATH'] + r"C:\Users\FischerVicto\AppData\Local\Programs\MiKTeX\miktex\bin\x64\mgs.exe"
    os.environ['PATH'] = os.environ['PATH'] + r"C:\Users\FischerVicto\AppData\Local\Programs\MiKTeX\miktex\bin\x64\mgsdll64.dll"
    os.environ['PATH'] = os.environ['PATH'] + "C:/Users/FischerVicto/AppData/Local/Programs/MiKTeX/"
    os.environ['PATH'] = os.environ['PATH'] + r"C:\Users\FischerVicto\AppData\Local\Programs\MiKTeX"

    os.environ['PATH'] = os.environ['PATH'] + r"C:/texlive/2023/bin/windows/"
    os.environ['PATH'] = os.environ['PATH'] + r"C:\Program Files\texstudio"
    os.environ['PATH'] = os.environ['PATH'] + "C:/Program Files/texstudio/"

    mpl.rcParams.update(mpl.rcParamsDefault)

    # for latex, you might need to change this
    label_size = 8
    # plt.style.use('fivethirtyeight')
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r'\usepackage{dsfont}\usepackage{amsmath}\usepackage{physics}')

    pr = {"axes.labelsize": 12,  # LaTeX default is 10pt font.
          "font.size": 12,
          "legend.fontsize": 12,  # Make the legend/label fonts
          "xtick.labelsize": 12,  # a little smaller
          "ytick.labelsize": 12,
          'figure.figsize': (width, height),
          "errorbar.capsize": 2.5,
          "font.family": "serif",
          "font.serif": [],  # blank entries should cause plots
          "font.sans-serif": [],
          }
    for k, v in pr.items():
        mpl.rcParams[k] = v

    mpl.rcParams["font.family"] = "serif"
    # mpl.rcParams["font.serif"] = ["STIX"]
    mpl.rcParams["mathtext.fontset"] = "stix"
