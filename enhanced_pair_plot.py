import numpy as np
import pandas as pd

import warnings

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats
import sklearn.feature_selection



# vertical gap to leave between annotations
line_height = 0.07

# properties for annotation bounding boxes 
# - idea is to slightly dim anything plotted underneath so that the annotation is clearer
annotation_bbox = dict(boxstyle='square,pad=0.02', facecolor='white', edgecolor='none', alpha=0.6, )
annotation_properties = dict(bbox=annotation_bbox, fontsize='x-small', )

def try_function(func, kwargs):
    try:
        return func(**kwargs)
    except BaseException as e:
        warnings.warn('While evaluating function ' + str(func) + ' on arguments ' + str(kwargs) + ', '
                      + 'the following error was raised: ' + str(e))
        return np.nan


def annotate_mean_and_sd(x, color, label=None, all_labels=[], **kwargs):  
    '''annotates the plot with the mean and SD of the data, 
    and draws vertical lines to indicate mean and interquartile range.
    
    all_labels is a list of all the hues, so that when this function is called for a specific hue,
    it can calculate the position of this hue in the list of hues, and position the annotation accordingly
    (so that the annotations aren't all overlapping with each other.)
    '''
    ax=plt.gca()
    # print(ax, color, label, type(label), dir(label), kwargs)

    if len(all_labels)>1:
        position_in_list_of_labels = [i for i in range(len(all_labels)) if all_labels[i]==label] [0]
    else: position_in_list_of_labels=0

    mean = try_function(func=np.mean, kwargs=dict(a=x))
    std =  try_function(func=np.std,  kwargs=dict(a=x))
    
    if not np.isnan([mean, std]).any(): 

        annotation_text = ('μ='+str(round(np.mean(x), 2)) + ', '
                            + 'σ='+str(round(np.std(x), 2)) )

        ax.annotate(annotation_text,   
                    xy=(0.03, 0.98 - line_height*position_in_list_of_labels),  
                    color=(color if (label!=None and label!='_nolegend_') else 'black'),
                    va='top', xycoords=ax.transAxes, **annotation_properties)

        ax.axvline(np.mean(x), alpha=.2, color=color)
        ax.axvline(np.quantile(x, .25,), alpha=.2, color=color, ls='dotted')
        ax.axvline(np.quantile(x, .75,), alpha=.2, color=color, ls='dotted')
        # ax.vlines(np.quantile(x, [.25,.75]), alpha=.15, colors=color, linestyles='dotted')




def annotate_correlation_coef(x, y, color, label=None, all_labels=[], **kwargs):    
    '''annotates the plot with the correlation coefficient, 
                and the p-val of whether there is a correlation.

    all_labels is a list of all the hues, so that when this function is called for a specific hue,
    it can calculate the position of this hue in the list of hues, and position the annotation accordingly
    (so that the annotations aren't all overlapping with each other.)
    '''
    ax=plt.gca()
    # print(ax, color, label, kwargs)

    if len(all_labels)>1:
        position_in_list_of_labels = [i for i in range(len(all_labels)) if all_labels[i]==label] [0]
    else: position_in_list_of_labels=0

    if label != 'outlier':
        if len(x)>1:

            rho, p_val = try_function(func=scipy.stats.pearsonr,
                                      kwargs=dict(x=x, y=y, alternative='two-sided'))

            if not np.isnan([rho, p_val]).any(): 
                annotation_text = ('ρ='+str(round(rho, 3)).replace('-', '−')    #make negative sign noticeable! − – — 
                                    + ', p-val='+str(round(p_val, 3))    )

                ax.annotate(annotation_text,   
                            xy=(0.03, 0.98 - line_height*position_in_list_of_labels),  
                            color=(color if (label!=None and label!='_nolegend_') else 'black'),
                            va='top',  xycoords=ax.transAxes,  **annotation_properties)




def annotate_mutual_information(x, y, color, label=None, all_labels=[], **kwargs):    
    '''annotates the plot with the mutual information.

    all_labels is a list of all the hues, so that when this function is called for a specific hue,
    it can calculate the position of this hue in the list of hues, and position the annotation accordingly
    (so that the annotations aren't all overlapping with each other.)
    '''
    ax=plt.gca()

    if len(all_labels)>1:
        position_in_list_of_labels = [i for i in range(len(all_labels)) if all_labels[i]==label] [0]
    else: position_in_list_of_labels=0

    if label != 'outlier':
        if len(x)>1:

            # MI = sklearn.feature_selection.mutual_info_regression(X=pd.DataFrame(x), y=y,
            #                                     discrete_features=False, random_state=0)

            MI = try_function(func=sklearn.feature_selection.mutual_info_regression,
                              kwargs=dict(X=pd.DataFrame(x), y=y,
                                         discrete_features=False, random_state=0))
            
            if not np.isnan(MI): 
                annotation_text = 'MI='+str(round(MI[0], 5))

                ax.annotate(annotation_text,   
                            xy=(0.03, 0.98 - line_height*position_in_list_of_labels),  
                            color=(color if (label!=None and label!='_nolegend_') else 'black'),
                            va='top', xycoords=ax.transAxes,   **annotation_properties)




def annotate_KS_test(x, color, label=None, all_labels=[], **kwargs):    
    '''annotates the plot with the p-value of the kolmogorov-smirnov test for normality.
    
    all_labels is a list of all the hues, so that when this function is called for a specific hue,
    it can calculate the position of this hue in the list of hues, and position the annotation accordingly
    (so that the annotations aren't all overlapping with each other.)
    '''
    ax=plt.gca()

    # first, standardise the data - otherwise, after removing outliers the data won't be standard, 
    #   and can cause the KS test to reject because of that.
    #   We want to see if it fails the KS test because of its *shape*, not because normalisation has been broken.
    # x_standardised = sklearn.preprocessing.StandardScaler(
    #     ).fit_transform(pd.DataFrame(x))
    x_standardised = (x-x.mean())/x.std()

    if len(all_labels)>1:
        position_in_list_of_labels = [i for i in range(len(all_labels)) if all_labels[i]==label] [0]
    else: position_in_list_of_labels=0

    if label != 'outlier':
        # KS_p_val = scipy.stats.ks_1samp(x_standardised, alternative='two-sided', cdf=scipy.stats.norm.cdf).pvalue
        KS = try_function(func=scipy.stats.ks_1samp,
                    kwargs=dict(x=x_standardised, alternative='two-sided', cdf=scipy.stats.norm.cdf))

        if not np.isnan(KS).any():
            ax.annotate('KS p-val='+str(round(KS.pvalue, 3)),   
                        xy=(0.03, 0.02 + max(0, line_height*(len(all_labels)-1)) - line_height*position_in_list_of_labels),  
                        color=(color if (label!=None and label!='_nolegend_') else 'black'),
                        va='bottom', xycoords=ax.transAxes,   **annotation_properties)






def selective_regplot(x, y, color, label=None, **kwargs):
    '''draws a linear fit to the data - unless it is outliers, in which case their shouldn't be such a line.'''
    ylim = plt.gca().get_ylim()
    data = pd.concat([x,y], axis=1)
    if label!='outlier':
        sns.regplot(data, x=x.name, y=y.name,
                    color=color,label=label, 
                    fit_reg=True, order=1, 
                    scatter=False, **kwargs)     # scatter is already done, don't redo

    plt.gca().set_ylim(ylim)            #reset ylim so that confidence regions don't distort it




def selective_kdeplot(x, y, color, label=None, **kwargs):
    '''plots a 2d kde plot for data - unless it is outliers, in which case their distribution shouldn't be plotted.'''
    # ylim = plt.gca().get_ylim()
    data = pd.concat([x,y], axis=1)
    if label!='outlier':
        sns.kdeplot(data, x=x.name, y=y.name,
                    color=color, #label=label, 
                    levels=7, **kwargs)

    # plt.gca().set_ylim(ylim)            #reset ylim so that contours don't distort it




def highlight_scatter(x, y, color, label=None, **kwargs):
    '''makes markers which stick out for outliers'''
    data = pd.concat([x,y], axis=1)
    if label=='outlier':
        sns.scatterplot(data, x=x.name, y=y.name,
                    color=color,label=label, 
                    # next line changes markers for each outlier. use index to ensure markers consistent all plots
                    style=data.index, 
                    s=120, linewidths=0, alpha=1)




def enhanced_pair_plot(df, hue=None, annotate_plots=True, show_N_on_legend=True):
    '''builds on the seaborn pairplot by 
    1. making a regression plot in one triangle and a kde 2d plot in the other
    2. displaying useful stats on each subplot:
        - diagonal:     a) mean&SD for each feature 
                        b) KS p-value evaluating whether it is from a normal distribution
        - triangles:    a) correlation coefficient between each pair of variables
                        b) p-value of statistical test testing null hypothesis that there is no correlation
                        c) mutual information between two variables.
        
        When there are two hues (e.g. inlier/outlier, test/train), 
            displays these stats for both when appropriate, in matching colour;
            if only one hue, displays the stats in black.
    '''
    
    if hue!=None and (np.unique(df[hue], return_counts=True)[1]).min()<30:
        warnings.simplefilter(action='ignore', category=UserWarning,)
    
    if show_N_on_legend and hue!=None:
        class_names, class_counts = np.unique(df[hue], return_counts=True)
        map={class_name:class_name + ' (N=' + str(class_count) + ')' 
                for (class_name, class_count) in zip(class_names, class_counts)}
        
        # convoluted way of changing hue column to avoid inane warnings stating 
        #   `A value is trying to be set on a copy of a slice from a DataFrame.`
        new_hue_col = df[hue].map(map)
        df = df.drop(columns=hue)
        df[hue] = new_hue_col


    # make scatters in off-diag, and kde in diag.
    plot = sns.pairplot(df, diag_kind='kde', plot_kws=dict(alpha=1, s=8), hue=hue,)
    
    # make regression line in top triangle, and 2-var kde in lower triangle
    plot.map_upper(selective_regplot, line_kws=dict(alpha=0.5))     
    plot.map_lower(selective_kdeplot, alpha=0.5)
    
    # highlight scatters for any points marked as outliers
    plot.map_offdiag(highlight_scatter,  )

    # we need a list of all the hues to pass to the annotation functions
    # so that annotations for each hue can be positioned correctly.
    # We also need it for ncols of legend.
    all_labels = ([] if hue==None else pd.unique(df[hue]))

    if annotate_plots:


        # annotate mean and SD of dist on diagonals
        plot.map_diag(annotate_mean_and_sd, all_labels=all_labels)

        # annotate KS test on the diagonals (tests that the distribution is normal)
        plot.map_diag(annotate_KS_test, all_labels=all_labels, )

        # annotate correlation coeff on upper triangle (with regression plot) 
        plot.map_upper(annotate_correlation_coef, all_labels=all_labels)

        # annotate MI on lower triangle (with KDE plot) 
        plot.map_lower(annotate_mutual_information, all_labels=all_labels)


    # if there is a legend, move it below the plot
    try:
        sns.move_legend(plot, loc='upper center', bbox_to_anchor=(0.5, 0), 
                        ncol=min(5, len(all_labels)))
        # as the legend has moved from the right, we use plt.tight_layout() to move the subplots into that empty space.
        plt.tight_layout()      

    except ValueError:
        pass

    plt.show()

    warnings.simplefilter(action='default', category=UserWarning,)  #reset










if __name__ == '__main__':
    # to compare to plots using normal seaborn.pairplot 
    #   at https://seaborn.pydata.org/generated/seaborn.pairplot.html
    penguins = sns.load_dataset('penguins')
    enhanced_pair_plot(penguins.dropna(), hue='species', annotate_plots=True)
