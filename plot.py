import plotly.tools 
import plotly.plotly as py
import plotly.io as pio
import plotly.graph_objs as go
import plotly.offline as pyoff
import os
import igraph as ig
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_PATH = "./plots/"
plotly.tools.set_credentials_file(username='isonettv', api_key='2Lg1USMkZAHONqo82eMG')


class vertexplotOpt(object):
    
    DEFAULT_CONSTANT_COLOR = (255*np.array([0,0.5,0])).astype(np.int64)
    
    def __init__(self,Y, mode = "discrete", palette = None, size = 1.5):
        self.mode = mode
        self.size = size
        
        if palette == None:
            if mode == "discrete":
                palette = "bright"
            else:
                palette = "coolwarm"
        
        if mode == "discrete":
            self.color_var  = color_scale_discrete(Y, palette)
            self.color_scale = None
            self.values = Y
            self.group_var = np.array(Y)
                
            self.names = vertex_name(Y)
        elif mode == "continuous":
            self.color_var  = np.array(color_scale_continuous(Y, palette))
            self.color_scale = color_scale_continuous(np.linspace(0,1,10), palette)
            self.color_scale = [[y,"rgb"+str(tuple(x[0:3]))] for x,y in zip(self.color_scale,np.linspace(0,1,10))]
            self.values = Y
            self.names = [str(x) for x in Y]
        elif mode == "constant":
            self.color_var  = np.repeat([vertexplotOpt.DEFAULT_CONSTANT_COLOR],len(Y),axis=0)
            self.color_scale = None
            self.values = Y
            

def plotGraph(X,W=None,vertex_opt = None,\
               plot_filename = str(datetime.datetime.now()) + ".png", online = False,\
              title = "", plotsize = [1000,1000], preprocessing = None, plot_dim = 2,\
            edge_width = 0.5, edge_color="black"):
    
    
        if plot_dim < 2 or plot_dim > 3:
            raise ValueError("plot_dim must be either 2 or 3")    
    
        if plot_dim > X.shape[1]:
            #Add missing dimensions
            temp = np.zeros((X.shape[0],plot_dim))
            temp[0:X.shape[0],0:X.shape[1]] = X
            X = temp
        
        if not os.path.exists(PLOT_PATH): 
            os.mkdir(PLOT_PATH)
        OUTPATH = PLOT_PATH + plot_filename
        
        
        
     
        def axis(dim_num):
            M = np.max(np.abs(X[:,dim_num]))        
            axis=dict(showbackground=False,
                      showline=True,
                      zeroline=False,
                      showgrid=False,
                      showticklabels=True,
                      title='',
                      range = [-(M+1),(M+1)]
                      )
            return(axis)
        
        scene = {}
        scene["xaxis"] = dict(axis(0))
        scene["yaxis"] = dict(axis(1))
        if plot_dim == 3:
            scene["zaxis"] = dict(axis(2))
         
        
        layout = go.Layout(
                 title=title,
                 width=plotsize[0],
                 height=plotsize[1],
                 showlegend=True,
                 scene=scene,
             margin=dict(
                t=100
            ),
            hovermode='closest',
            annotations=[
                   dict(
                   showarrow=False,
                    text="Data source:</a>",
                    xref='paper',
                    yref='paper',
                    x=0,
                    y=0.1,
                    xanchor='left',
                    yanchor='bottom',
                    font=dict(
                    size=14
                    )
                    )
                ],    )     
        
        #Do preprocessing
        if preprocessing != None:
            preprocessing = preprocessing.lower()
            if preprocessing == "pca":
                raise NotImplemented()
            elif preprocessing == "lle":
                raise NotImplemented()
            else:
                raise ValueError("Preprocessing type not found.")
        
        

         
        
        #Create Traces
        data = []
        
        if not W is None:
            trace_edge = traceEdges(X=X, W=W,plot_dim=plot_dim, edge_width=edge_width,
                        edge_color=edge_color)
            data = [trace_edge]
            
        trace_vertex = traceVertex(X=X, plot_dim=plot_dim, v_opt = vertex_opt)
        data += trace_vertex
        
        #Create figure
        fig=go.Figure(data=data, layout=layout)
        
        
        print("Plotting graph...")
        if online:
            try:
                py.iplot(fig)
            except plotly.exceptions.PlotlyRequestError:
                print("Warning: Could not plot online")
                
        #pyoff.offline.plot(fig)
        pio.write_image(fig,OUTPATH)
        print("Done!")  



def traceVertex(X,plot_dim, v_opt):
        if X.shape[1] != plot_dim:
            raise ValueError("data does not have dimension equal to the one specified by the plot")
        
        X = np.array(X)
        trace = []
        
        def flatten(X):
            return np.reshape(X,(-1))
        
        ''' Creates the style attribute for discrete plot'''
        def getStyle(values,color_var):
            styles = {}
            for i, x in enumerate(values):
                styles[x] = color_var[i]
            return [dict(target = x, value = dict(marker = dict(color = y))) \
                    for x,y in styles.items()]
            
        
        if v_opt.mode == "discrete":
            #One plot for each group
            levels = np.unique(v_opt.group_var)
            call_dicts = []
            for l in levels:
                l_ids = np.where(v_opt.group_var == l)
                l_dict =\
                    dict(
                       x=flatten(X[l_ids,0]),
                       y=flatten(X[l_ids,1]),
                       z = None if plot_dim == 2 else flatten(X[l_ids,2]),
                       name=str(l),
                       mode='markers',
                       marker=dict(symbol='circle',
                                     size=v_opt.size,
                                     color=v_opt.color_var[l_ids],
                                     line=dict(color='rgb(50,50,50)', width=0.1),
                                     ),
                       hoverinfo='text')
                call_dicts += [l_dict]
            
            
        else:
            call_dicts = [\
                dict(  x=X[:,0],
                       y=X[:,1],
                       z=None if plot_dim == 2 else X[:,2],
                       mode='markers',
                       showlegend = True,
                       marker=dict(symbol='circle',
                                     size=v_opt.size,
                                     color=v_opt.color_var,
                                     line=dict(color='rgb(50,50,50)', width=0.1),
                                     ),
                       hoverinfo='text')]
        
            
        for call_dict in call_dicts:   
             
            if plot_dim == 2:
                call_dict.pop("z")
                call_dict["marker"]["color"] = ["rgb"+str(tuple(x)) for x in call_dict["marker"]["color"]]
                call_dict["text"] = [str(x) for x in v_opt.values]
                call_dict["type"] = "scattergl"    
            else:
                call_dict["text"] = v_opt.values
                call_dict["type"] = "scatter3d"
                
                
            
            if v_opt.mode == "continuous":
                #Add color scale
                call_dict["marker"]["colorscale"] = v_opt.color_scale 
                print(v_opt.color_scale)
                call_dict["marker"]["cmax"] = np.max(v_opt.values)
                call_dict["marker"]["cmin"] = np.min(v_opt.values)
                call_dict["marker"]["colorbar"] = dict(title="scale")
                
                

            
        return(call_dicts)
        


def traceEdges(X,W,edge_color,plot_dim,edge_width = 0.0005):
        xe=[]
        ye=[]
        ze=[]
        
        for i in np.arange(X.shape[0]):

            for j in np.arange(X.shape[0]):
                
                if W[i,j] + W[j,i] == 0: continue    
            
                xe.extend([X[i,0],X[j,0]])# x-coordinates of edge ends
                ye.extend([X[i,1],X[j,1]])# y-coordinates of edge ends
                if plot_dim > 2:
                    ze.extend([X[i,2],X[j,2]])# z-coordinates of edge ends
            
        print(xe)
            
        #Create 2D or 3D trace
        if (plot_dim == 2):
            trace=dict(x=xe,
                       y=ye,
                       type="scatter",
                       mode='lines',
                       line=dict(color='rgb(210,210,210)',
                                  width=edge_width),
                       hoverinfo='none'
                       )
        else:
            trace=dict(x=xe,
                       y=ye,
                       z=ze,
                       type="scatter3d",
                       mode='lines',
                       line=dict(color=edge_color,
                                  width=edge_width),
                       hoverinfo='none'
                       )
        return(trace)
        
        
def getEdgeColorScale(W,edge_colorscale_range = ["0to1","relative","constant"][0]):
        ce = []

        for i in np.arange(W.shape[0]):
            for j in np.arange(W.shape[1]):
                if i < j: continue  
                ce += 2*[W[i,j] + W[j,i]] #color of edge:
                
  
        #Create Scaling for edges based on strength
        num_scales = 1000
        
        
        pal_E = ig.GradientPalette("white", "black", num_scales)
        ce_max = np.max(ce)
        ce_min = np.min(ce)
        if edge_colorscale_range == "0to1":
            if ce_max > 1 or ce_min < 0:
                raise ValueError("Edge strength falls outside [0,1] range")
            colors = [pal_E.get(int(np.round(num_scales*ce[k]))) for k,_ in enumerate(ce)]
        elif edge_colorscale_range == "relative":
            if ce_max == ce_min:
                colors = "black"
            else:
                colors = [pal_E.get(int(np.round(num_scales*(ce[k]-ce_min)/(ce_max-ce_min)))) for k,_ in enumerate(ce)]
        else:
            colors = "black"
            
        return (colors)
        
def color_scale_discrete(Y,palette="bright"):
    Y = Y - np.min(Y) 
    pal = sns.color_palette(palette,2)
    res = np.array(list(map(lambda k: (pal[int(k)]),Y)))
    print(res.shape)
    return(res)

def color_scale_continuous(Y,palette="coolwarm",num_palette=70):
    
    Y = (Y - np.min(Y))
    Y = Y / np.max(Y) 
    
    pal = ig.AdvancedGradientPalette(sns.color_palette(palette,7),n=num_palette+1)
    res = 255*np.array(list(map(lambda k: (pal.get(int(num_palette*k))),Y)))
    res = res.astype(np.int64)
    
    return res

    
def vertex_name(Y):
    return(["Class" + str(x) for x in Y])