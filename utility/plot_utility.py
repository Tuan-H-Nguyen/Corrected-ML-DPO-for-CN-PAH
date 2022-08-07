import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter

#plt font initialize
annotate = {'fontname':'Times New Roman','weight':'bold','size':13}
tick = {'fontname':'Times New Roman','size':13}
font = FontProperties()
font.set_weight('bold')
font_legend = font_manager.FontProperties(family = 'Times New Roman',size = 12)
  
def plot_histogram(
    dataset,
    save_path=None,
    label=None,
    label_loc=None,
    x_labels=True
    ):
    fig,ax = plt.subplots(nrows=1,ncols=1)
    ax.hist(
        dataset.loc[:]['Egap'],
        bins=[1.5,2,2.5,3,3.5,4,4.5,5]
        )
    ax.set_ylabel('Number of samples (samples)',**annotate)
    #
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
    #
    labely = ax.get_yticks().tolist()
    ax.yaxis.set_ticklabels(labely,**tick)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))  
    #
    if x_labels:
        ax.set_xlabel('Band gap (eV)',**annotate)
        labelx = ax.get_xticks().tolist()
        ax.xaxis.set_ticklabels(labelx,**tick)
    else:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_minor_locator(plt.NullLocator())

    if label != None and label_loc != None:
        x,y = label_loc
        ax.text(x,y,label,**annotate)
    if save_path != None:
        fig.savefig(save_path,dpi=600,bbox_inches='tight')

class scatter_plot:
    def __init__(self):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)

    def add_plot(
        self,
        x,y,
        xlabel,
        ylabel,
        weight=None,
        i=None,
        xticks_format=2,
        yticks_format=2,
        x_minor_tick=None,
        x_major_tick=None,
        y_minor_tick=None,
        y_major_tick=None,
        xlim=None, ylim=None,
        line_color = None,
        scatter_color = None,
        label =None,
        equal_aspect = False
        ):
        self.ax.scatter(x,y,c = scatter_color, label = label,s=15)
        if weight == None:
            self.ax.plot(x,y,linewidth=1.5, c=line_color)
        else:
            assert isinstance(weight,tuple)
            wb,w = weight
            if i == None:
                i = np.linspace(min(x),max(x),1000)
            else:
                assert isinstance(i,tuple)
                i = np.linspace(i[0],i[1],100)
            self.ax.plot(i,wb+w*i,linewidth=1.5, c=line_color)
            #
        if equal_aspect:
            self.fig.gca().set_aspect('equal',adjustable='box')
        #
        self.ax.set_xlabel(xlabel,**annotate)
        if xlim:
            x,y = xlim
            self.ax.set_xlim(x,y)
        if type(x_major_tick) is float or type(x_major_tick) is int:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(x_major_tick))
        elif x_major_tick == 'null':
            self.ax.xaxis.set_major_locator(plt.NullLocator())
        if x_minor_tick:
            self.ax.xaxis.set_minor_locator(plt.MultipleLocator(x_minor_tick))
        try:
            labelx = self.ax.get_xticks().tolist()
            self.ax.xaxis.set_ticklabels(labelx,**tick)
            xticks_format = '%.f' if xticks_format==0 else '%.'+str(xticks_format)+'f'
            self.ax.xaxis.set_major_formatter(FormatStrFormatter(xticks_format))
        except AttributeError:
            pass
            #
        self.ax.set_ylabel(ylabel,**annotate)
        if ylim:
            x,y = ylim
            self.ax.set_ylim(x,y)
        if type(y_major_tick) is float or type(y_major_tick) is int:
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(y_major_tick))
        elif y_major_tick == 'null':
            self.ax.yaxis.set_major_locator(plt.NullLocator())
        if y_minor_tick:
            self.ax.yaxis.set_minor_locator(plt.MultipleLocator(y_minor_tick))
        labely = self.ax.get_yticks().tolist()
        self.ax.yaxis.set_ticklabels(labely,**tick)
        yticks_format = '%.f' if yticks_format==0 else '%.'+str(yticks_format)+'f'
        self.ax.yaxis.set_major_formatter(FormatStrFormatter(yticks_format))
    
    def add_text(self,x,y,text):
        self.ax.text(x,y,text,**annotate) 

    def add_legend(self):
        self.ax.legend(prop = font_legend)

    def save_fig(self,save_path,dpi=600):
        #self.ax.legend()
        self.fig.savefig(save_path,dpi=dpi,bbox_inches="tight")

    def clear(self):
        self.fig.clf()
        del self.fig
