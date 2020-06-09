"""
GP_flowsheet project

                          Ploting utilities

Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
Date: April 2020
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from matplotlib.animation import FuncAnimation

class plot():
    
    ########################
    # --- Initializing --- #
    ########################
    def __init__(self, _3D=False):
        self.plt = plt
        self.fig = plt.figure(figsize=(8,5))
        self.left, self.bottom, self.width, self.height = 0.1, 0.1, 0.8, 0.8
        self.ax = self.fig.add_axes([self.left, self.bottom, self.width, self.height])
        if _3D:
            self.ax3d = Axes3D(self.fig)
    
    ########################    
    # --- Contour plot --- #
    ########################
    
    def contour(self, X, Y, Z, title='Title', xlabel='x', ylabel='y'):
        
        # -- Internal variables
        ax  = self.ax
        # -- Plot
        cp = ax.contour(X, Y, Z)
        ax.clabel(cp, inline=True, fontsize=9)
        ax.set_title(title, size=15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
    ########################    
    # --- Scatter plot --- #
    ########################
    def scatter(self, X, Y, label='Scatter label', color='k', marker='x', alpha=1, text=[], markersize=30):
        # -- Internal variables
        ax = self.ax
        # -- Plot
        ax.scatter(X, Y, color=color, marker=marker, label=label, alpha=alpha, s=markersize)
        if len(text) > 0:
            for i, txt in enumerate(text):
                ax.annotate(txt, (X[i] + 3e-2, Y[i] - 5e-2))
        
    ########################    
    # --- Show plot --- #
    ########################
    def show(self, legend_b=True):
        if legend_b:
            self.plt.legend(loc='lower center')
        return self.plt.show
    
    ###########################    
    # --- 3D surface plot --- #
    ###########################
    def surface_3D(self, X, Y, Z, title='Rosenbrock function', xlabel='x', ylabel='y', zlabel='z'):
        # -- Internal variables
        ax   = self.ax3d
        ax.plot_surface(X, Y, Z, cmap=cm.YlGnBu)
        ax.set_title(title, size=15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        
    ###########################    
    # --- 3D scatter plot --- #
    ###########################
    def scatter_3D(self, X, Y, Z, title='Rosenbrock function', xlabel='x', ylabel='y', zlabel='z'):
        # -- Internal variables
        ax   = self.ax3d
        ax.scatter(X, Y, Z)
        ax.set_title(title, size=15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        
    ###################    
    # --- TR plot --- #
    ###################
    def TR_plot(self, center, radius, color='k'):
        # -- Internal variables
        ax   = self.ax
        TR   = self.plt.Circle(center, radius, color=color, fill=False, linestyle='--', linewidth=0.8)
        ax.add_artist(TR)
        
    #####################################    
    # --- Set xlim and ylim of plot --- #
    #####################################
    def xylim(self, xlim_, ylim_):
        self.plt.xlim(xlim_)
        self.plt.ylim(ylim_)
    
    #####################    
    # --- Line plot --- #
    #####################
    def line(self, X, Y, label='Line label', color='k', ls='-'):
        # -- Internal variables
        ax = self.ax
        # -- Plot
        ax.plot(X, Y, color=color, label=label, ls=ls)
    
    #########################    
    # --- GIF animation --- #
    #########################
    def gif(self, positions, d, radii, X, Y, Z, const=None, title='Title', xlabel='x', ylabel='y', name='name.gif'):
        # -- Internal variables
        fig = self.fig
        plt = self.plt
        ax  = self.ax
        
        def TR_plot(plt, ax, center, radius):
            TR   = plt.Circle(center, radius, color='k', fill=False, linestyle='--', linewidth=0.8)
            ax.add_artist(TR)
        
        if const != None:
            for c in const:
                if len(c) == 2:
                    ax.plot(c[0], c[1], color='k')
                else:
                    ax.plot(c[0], c[1], color=c[2], ls=c[3])
        
        cp = ax.contour(X, Y, Z)
        ax.clabel(cp, inline=True, fontsize=9)
        ax.set_title(title, size=15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        optima             = positions + d
        points             = np.zeros(1, dtype=[("position", float, 2)])
        points["position"] = optima[0]
        scatter            = ax.scatter(points["position"][:,0], points["position"][:,1], c='blue', s=10)
        
        P_radius           = np.append(positions, radii.reshape(-1,1), axis=1)
        
        def update(frame_number):
            points["position"] = optima[frame_number]
            scatter.set_offsets(points["position"])
            
            center = (P_radius[frame_number,0], P_radius[frame_number,1])
            r      = P_radius[frame_number,2]
            TR     = TR_plot(plt, ax, center, r)
            return scatter, TR,
    
        anim = FuncAnimation(fig, update, interval=200, 
                             frames=range(len(positions)), repeat_delay=2000)
        
        # Save gif
        anim.save(name, writer='ffmpeg', fps=2)
        
        
def meshgrid(function, lb, ub):
    '''
    Creates a meshgrid of a given function

    Parameters
    ----------
    function : Function
        Function to evaluate the meshgrid.
    lb : np.array
        Lower bounds array.
    ub : np.array
        Upper bounds array.

    Returns
    -------
    X : np.array
        Meshgrid X values.
    Y : np.array
        Meshgrid Y values.
    Z : np.array
        Meshgrid Z values (function evaluations).
    '''
    x       = np.linspace(lb[0], ub[0], 100)
    y       = np.linspace(lb[1], ub[1], 100)
    X, Y    = np.meshgrid(x, y)
    Z       = function(X, Y)
    return X, Y, Z
    

