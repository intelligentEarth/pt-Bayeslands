from __future__ import print_function, division

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import time
import operator
import math
import cmocean as cmo
from pylab import rcParams
import copy
from copy import deepcopy
import cmocean as cmo
from pylab import rcParams
import collections
from scipy import special
import fnmatch
import shutil
from PIL import Image
from io import StringIO
from cycler import cycler
import os
import sys
import matplotlib.mlab as mlab
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats 
from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import itertools
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()
from plotly.offline.offline import _plot_html
import pandas
import argparse


def viewGrid( width=1000, height=1000, zmin=None, zmax=None, zData=None, title='Predicted Topography', time_frame=None, filename=None):

    #https://plot.ly/python/reference/#layout-yaxis-title-font-size
    #https://plot.ly/r/reference/#heatmap-showscale

        if zmin == None:
            zmin =  zData.min()

        if zmax == None:
            zmax =  zData.max()

        tickvals= [0,50,75,-50]

        xx = (np.linspace(0, zData.shape[0] , num=zData.shape[0]/10 )) 
        yy = (np.linspace(0, zData.shape[1]  , num=zData.shape[1]/10 )) 

        xx = np.around(xx, decimals=0)
        yy = np.around(yy, decimals=0)
        print (xx,' xx')
        print (yy,' yy')

        # range = [0,zData.shape[0]* self.resolu_factor]
        #range = [0,zData.shape[1]* self.resolu_factor],

        axislabelsize = 20

        data = Data([Surface(x= zData.shape[0] , y= zData.shape[1] , z=zData, colorscale='YlGnBu')])

        #https://plot.ly/r/reference/#scatter3d 

        # try hide colour bar - todo 

        layout = Layout(    autosize=False, width=width, height=height,scene=Scene(
                    zaxis=ZAxis(title = 'Elev. (m)   ', range=[zmin,zmax], autorange=False, nticks=5, gridcolor='rgb(255, 255, 255)',
                                gridwidth=2,  zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True, titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ), tickmode="auto"),

                    xaxis=XAxis(title = 'x-axis  ',   tickvals= xx,   nticks=5 ,  gridcolor='rgb(255, 255, 255)', gridwidth=2,
                                zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True, titlefont=dict(size=axislabelsize) ,  tickfont=dict(size=14 ),  ),

                    yaxis=YAxis(title = 'y-axis  ', tickvals= yy,   nticks=5, gridcolor='rgb(255, 255, 255)', gridwidth=2,
                                zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True,  titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ), ),
                    bgcolor="rgb(244, 244, 248)" ))

        fig = Figure(data=data, layout=layout) 
        graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename= filename+'_.html', validate=False)

         


def main():

    real_elev = np.random.rand(100,100)


    viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=  real_elev , title='Ground truth Topography', time_frame= 10000, filename = 'ground_truth')
       



if __name__ == "__main__": main()


