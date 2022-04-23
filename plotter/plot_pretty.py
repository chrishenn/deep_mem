import matplotlib.pyplot as plt

import numpy as np




def plot_pretty(all_data, legends=None, ylabel='ylabel', xlabel='xlabel', ylim=None, xlim=None, **kwargs):
    '''
    param all_data is list of np.array
    param legend is list of strings, to mark each array of data in order
    param ylabel, xlabel are label strings
    param ylim, xlim are 2-item {}
    param kwargs is dictionary of plt styles, where 'key:val' gives
        'style param string : list of style option-strings, each selected in order to match each array in all_data'
        or, 'style param string : option-string to be used for all arrays in all_data'
    '''

    kwargs.setdefault('color', ['r','g','b','c','m','y','k'] )
    kwargs.setdefault('linestyle', 'solid')
    kwargs.setdefault('marker', [".","o","v","^","<",">","s","p","P","*","+","x","d"]  )
    kwargs.setdefault('markevery', 1)

    styles = [ { (k):(v if not isinstance(v, list) else v[i])  for k,v in kwargs.items() } for i in range(len(all_data))]

    for i in range(len(all_data)):
        arry = all_data[i]
        plt.plot(np.arange(len(arry))+1, arry, **styles[i] )

    # aesthetics
    ax_ = plt.gca() 
    ax_.spines['right'].set_color((.8,.8,.8))
    ax_.spines['top'].set_color((.8,.8,.8))           

    plt.grid(linestyle='--')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None: 
        plt.ylim(ylim)
        
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if legends is not None:
        plt.legend(legends) 
    plt.ion()



if __name__ == "__main__":
    all_data = [np.array([32,45,56,78]), np.array([22,35,46,68])]
    plot_pretty(all_data, ['train_acc', 'val_acc'], xlabel='Epoch', ylabel='Accuracy')

    print()
    

