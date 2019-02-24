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
import sklearn.preprocessing as skpp

PLOT_PATH = "./plots/"
plotly.tools.set_credentials_file(username='isonettv', api_key='2Lg1USMkZAHONqo82eMG')


class vertexplotOpt(object):
    
    DEFAULT_CONSTANT_COLOR = (255*np.array([0,0,0])).astype(np.int64)
    DEFAULT_UNLABELED_COLOR = (255*np.array([0,0.5,0])).astype(np.int64)
    
    
    def __init__(self,Y, mode = "discrete", palette = None, size = 1.5,\
                 labeledIndexes = None):
        self.mode = mode
        self.values = np.array(Y)
        
        if np.array(size).shape == ():
            self.size = np.repeat(size,Y.shape[0])
        
        if palette == None:
            if mode == "discrete":
                palette = "bright"
            else:
                palette = "coolwarm"
        
        if mode == "discrete":
            self.color_var  = color_scale_discrete(Y, palette)
            self.color_scale = None
            self.group_var = np.array(Y)
            self.names = vertex_name(Y)
        elif mode == "continuous":
            self.color_var  = np.array(color_scale_continuous(Y, palette))
            self.color_scale = color_scale_continuous(np.linspace(0,1,10), palette)
            self.color_scale = [[y,"rgb"+str(tuple(x[0:3]))] for x,y in zip(self.color_scale,np.linspace(0,1,10))]
            self.names = [str(x) for x in Y]
        elif mode == "constant":
            self.color_var  = np.repeat([vertexplotOpt.DEFAULT_CONSTANT_COLOR],len(Y),axis=0)
            self.color_scale = None
        if not labeledIndexes is None:
            self.color_var[np.logical_not(labeledIndexes)] = vertexplotOpt.DEFAULT_UNLABELED_COLOR
            self.values[np.logical_not(labeledIndexes)] = -1
            self.size[np.logical_not(labeledIndexes)] = 0.5*self.size[np.logical_not(labeledIndexes)]
            if mode == "discrete":
                self.group_var[np.logical_not(labeledIndexes)] = -1

def plotGraph(X,W=None,vertex_opt = None,plot_filename = None, online = False,
              interactive=False,title = "", plotsize = [1000,1000], preprocessing = None, 
            plot_dim = 2, edge_width = 0.5):
    
        if plot_filename is None:
            plot_filename = str(datetime.datetime.now()) + ".png"
    
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
            trace_edge = traceEdges(X=X, W=W,plot_dim=plot_dim, edge_width=edge_width)
            data = trace_edge
            
        trace_vertex = traceVertex(X=X, plot_dim=plot_dim, v_opt = vertex_opt)
        data += trace_vertex
        
        #Create figure
        fig=go.Figure(data=data, layout=layout)
        
        
        print("Plotting graph..." + title)
        if online:
            try:
                py.iplot(fig)
            except plotly.exceptions.PlotlyRequestError:
                print("Warning: Could not plot online")
                
        if interactive:
            pyoff.offline.plot(fig)
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
                       name=vertex_name([v_opt.values[l_ids][0]])[0],
                       text = vertex_name(v_opt.values[l_ids]),
                       mode='markers',
                       marker=dict(symbol='circle',
                                     size=v_opt.size[l_ids],
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
                                     size=v_opt.size[l_ids],
                                     color=v_opt.color_var,
                                     line=dict(color='rgb(50,50,50)', width=0.1),
                                     ),
                       hoverinfo='text')]
        
            
        for call_dict in call_dicts:   
             
            if plot_dim == 2:
                call_dict.pop("z")
                call_dict["marker"]["color"] = ["rgb"+str(tuple(x)) for x in call_dict["marker"]["color"]]
                #print(call_dict["marker"]["color"])
                call_dict["text"] = [str(x) for x in v_opt.values]
                call_dict["type"] = "scattergl"    
            else:
                call_dict["text"] = v_opt.values
                call_dict["type"] = "scatter3d"
                
                
            
            if v_opt.mode == "continuous":
                #Add color scale
                call_dict["marker"]["colorscale"] = v_opt.color_scale 
                call_dict["marker"]["cmax"] = np.max(v_opt.values)
                call_dict["marker"]["cmin"] = np.min(v_opt.values)
                call_dict["marker"]["colorbar"] = dict(title="scale")
                
                

            
        return(call_dicts)
        


def traceEdges(X,W,plot_dim,edge_width):
        xe=[]
        ye=[]
        ze=[]
        ce = []
        trace=[]
        
        def flatten(x):
            return(np.reshape(x,(-1)))
        
        for i in np.arange(X.shape[0]):

            for j in np.arange(X.shape[0]):
                temp = W[i,j] + W[j,i]
                if temp == 0: continue 
                 
            
                xe.append([X[i,0],X[j,0],None])# x-coordinates of edge ends
                ye.append([X[i,1],X[j,1],None])# y-coordinates of edge ends
                if plot_dim > 2:
                    ze.append([X[i,2],X[j,2],None])# z-coordinates of edge ends
                ce.append(0.5*temp)
            
        
        
        xe = np.array(xe)
        ye = np.array(ye)
        ze = np.array(ze)
        ce = np.array(ce)
        ids = np.argsort(ce)
        if np.max(ce) == np.min(ce):
            ce = np.linspace(0.5,0.5,ce.shape[0])
        else:
            ce = np.linspace(0,1,ce.shape[0])
        xe = xe[ids]
        ye = ye[ids]
        if plot_dim > 2:
            ze = ze[ids]
        
       
        
        splt = [list(x) for x in np.array_split(np.arange(len(ce)),10)]
        


        max_brightness = 210
        for x in splt:
            
            col = max_brightness - int(max_brightness*ce[x][0])
            col = 'rgb' + str( (col,col,col) ) if plot_dim == 2 else (col,col,col)
            new_edge = dict(x=flatten(xe[x]),
                       y=flatten(ye[x]),
                       type="scattergl",
                       mode='lines',
                       showlegend=False,
                       line=dict(color = col,
                                  width=edge_width),
                       hoverinfo='none'
                       )
            if plot_dim == 3:
                new_edge["z"] = flatten(ze[x])
                new_edge["type"] = "scatter3d"
            trace.append(new_edge)
            
        return(trace)
        
        

        
def color_scale_discrete(Y,palette="bright"):
    Y = Y - np.min(Y) 
    pal = sns.color_palette(palette,np.max(Y)+1)
    res = 255*np.array(list(map(lambda k: (pal[int(k)]),Y)))
    return(res)

def color_scale_continuous(Y,palette="coolwarm",num_palette=70):
    
    Y = (Y - np.min(Y))
    Y = Y / np.max(Y) 
    
    pal = ig.AdvancedGradientPalette(sns.color_palette(palette,7),n=num_palette+1)
    res = 255*np.array(list(map(lambda k: (pal.get(int(num_palette*k))),Y)))
    res = res.astype(np.int64)
    
    return res

    
def vertex_name(Y):
    return(["Class" + str(x) if x != -1 else "unlabeled" for x in Y])
    
