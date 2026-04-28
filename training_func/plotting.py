from sklearn.metrics import auc, precision_recall_curve, roc_curve
import math, sys
import matplotlib.pyplot as plt
import numpy as np
from training_func.utils import to_formal
import training_func.environment as env


# Draw the picturer of k-fold trainning results and output the filename of the picture
def draw_metric(image_path, histories, metric, title='Learning Curve'):

    epochs = len(histories[0][metric])
    x_ticks = np.arange(1, epochs + 1)

    top_value = 0.0
    
    properties = {metric: ('Training', 'steelblue', 'steelblue'),
                  'val_%s' % metric: ('Validation', 'coral', 'coral'),
                  'samp_val_%s' % metric: ('Sampled Validation', 'olivedrab', 'olivedrab')}

    plt.figure(figsize=(10, 5))

    for metric_key, (label_name, color, markercolor) in properties.items():

        if metric_key in histories[0]:

            stats = np.array([h[metric_key] for h in histories])
            min_stats = np.min(stats, axis=0)
            max_stats = np.max(stats, axis=0)
            mean_stats = np.mean(stats, axis=0)
    
            plt.plot(x_ticks, min_stats, '--', color=color, linewidth=1.5, alpha=0.4)
            plt.plot(x_ticks, max_stats, '--', color=color, linewidth=1.5, alpha=0.4)
            plt.fill_between(x_ticks, min_stats, max_stats, color=color, alpha=0.15)
            plt.plot(x_ticks, mean_stats, '-o', color=color, linewidth=2,
                     markeredgecolor=markercolor, markeredgewidth=1, markersize=3,
                     label=r'%d-Fold %s Average (%.3f $\pm$ %.3f)' % (env.K_FOLD, label_name, mean_stats[-1], stats.std(axis=0)[-1]))

            top_value = max(top_value, np.max(max_stats) * 1.5)

    plt.grid(color='gainsboro', linestyle='dotted', linewidth=1, alpha=0.3)
    plt.xticks(np.r_[np.arange(1, epochs, math.ceil(epochs / 10)), epochs], fontsize=13)

    if metric == 'loss':
        plt.ylim(0.0, top_value)
        plt.yticks(np.around(np.linspace(0.0, top_value, 6), 3), fontsize=13)
    else:
        plt.ylim(0.0, 1.0)
        plt.yticks(np.linspace(0.0, 1.0, 6), fontsize=12)
        plt.gca().set_yticklabels(['%d%%' % (_ * 100) for _ in plt.gca().get_yticks()])

    plt.title(title, fontsize=16, pad=10)
    plt.xlabel('Epoch', fontsize=16, labelpad=10)
    plt.ylabel(to_formal(metric), fontsize=16, labelpad=10)
    plt.legend(fontsize=12)
    plt.tight_layout(pad=2)

    plt.savefig(image_path, dpi=300)
    plt.close()


# Draw ROC or PR curve and calculate area
def draw_curve(image_path, predictions, curve_type, labels=None, title='Learning Curve'):

    y_interps = list()
    areas = list()
    baseline = np.linspace(0, 1, 3500)
    plt.figure(figsize=(8, 8))

    if curve_type == 'ROC':
        plt.plot([0, 1], [0, 1], '--', color='gainsboro', linewidth=1)

    # Calculate & interpolate
    for prediction in predictions:

        if curve_type == 'ROC':
            fpr, tpr, _ = roc_curve(prediction['y_true'], prediction['y_pred'])
            y_interp = np.interp(baseline, fpr, tpr)
        elif curve_type == 'PRC':
            precision, recall, _ = precision_recall_curve(prediction['y_true'], prediction['y_pred'])
            y_interp = np.interp(baseline, recall[::-1], precision[::-1])

        y_interps.append(y_interp)
        areas.append(auc(baseline, y_interp))
    
    if labels is None:

        lower_y_interp = np.min(y_interps, axis=0)
        upper_y_interp = np.max(y_interps, axis=0)
        mean_y_interp = np.mean(y_interps, axis=0) 

        plt.plot(baseline, lower_y_interp, '--', color='sandybrown', linewidth=1.5, alpha=0.4)
        plt.plot(baseline, upper_y_interp, '--', color='sandybrown', linewidth=1.5, alpha=0.4)
        plt.fill_between(baseline, lower_y_interp, upper_y_interp, color='sandybrown', alpha=0.15)
        plt.plot(baseline, mean_y_interp, color='salmon', linewidth=2,
                 label=r'%d-Fold Validation Average (au%s = %.3f $\pm$ %.3f)' % (env.K_FOLD, curve_type, auc(baseline, mean_y_interp), np.std(areas)))

    else:

        colors = plt.cm.get_cmap('tab10').colors + plt.cm.get_cmap('Accent').colors
        for i in range(len(predictions)):
            plt.plot(baseline, y_interps[i], color=colors[i], zorder=-i,
                     label='%s (au%s = %.3f)' % (labels[i], curve_type, areas[i]))
    
    # Some specific edit about the roc picture
    if curve_type == 'ROC':
        plt.xlabel('1 - Specificity', fontsize=16, labelpad=10)
        plt.ylabel('Sensitivity', fontsize=16, labelpad=10)
    elif curve_type == 'PRC':
        plt.xlabel('Recall', fontsize=16, labelpad=10)
        plt.ylabel('Precision', fontsize=16, labelpad=10)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xticks(np.arange(0.0, 1.2, 0.2), fontsize=14)
    plt.yticks(np.arange(0.2, 1.2, 0.2), fontsize=14)
    plt.gca().set_xticklabels(['%d%%' % (_ * 100) for _ in plt.gca().get_xticks()])
    plt.gca().set_yticklabels(['%d%%' % (_ * 100) for _ in plt.gca().get_yticks()])

    plt.title(title, fontsize=14, pad=10)
    plt.legend(fontsize=13)
    plt.tight_layout(pad=2)

    plt.savefig(image_path, dpi=300)
    plt.close()




def write_prob_txt(la_li, prb_li, t2_prob_txt):
    if len(la_li) != len(prb_li): 
        print( len(la_li), len(prb_li), '### Error: 2 list len mismatch ')
        sys.exit()

    with open( t2_prob_txt , 'w') as op:
        for i,v in enumerate(la_li):
            #1_0.798
            op.write( str(v)+'_'+str( round(prb_li[i],3) ) +'\n' )
    #print('== op_test_label_prob complete ==')



# show_table return string
def show_table(values, headers=None, v_headers=None, title=None, float_fmt='%.3f'):
    txt=''
    values = [list(_) for _ in values]

    if headers is not None:
        headers = list(headers)
        item_widths = [len(_) for _ in headers]
    else:
        item_widths = [0 for _ in range(len(values[0]))]

    if v_headers is not None:
        if headers is not None:
            headers.insert(0, '')
        for i, row in enumerate(values):
            row.insert(0, v_headers[i])
        item_widths.insert(0, 0)
    
    for row in values:
        for i, v in enumerate(row):
            row[i] = float_fmt % v if isinstance(v, float) else str(v)
            item_widths[i] = max(item_widths[i], len(row[i]))

    sep_line = '+%s+' % '+'.join('-' * (w + 2) for w in item_widths)

    if title is not None:
        print('+%s+' % ('-' * (len(sep_line) - 2)))
        print('| %%-%ds |' % (len(sep_line) - 4) % title)
        txt = txt + ('+%s+' % ('-' * (len(sep_line) - 2))) +'\n'
        txt = txt + ('| %%-%ds |' % (len(sep_line) - 4) % title) +'\n'
    
    print(sep_line)
    txt = txt + sep_line +'\n'

    if headers is not None:
        #print(headers) #['', 'Acc', 'Spec', 'Prec', 'Recall', 'f1', 'MCC', 'auROC', 'auPRC']
        #print(item_widths) #[0, 5, 5, 5, 6, 5, 5, 5, 5]
        for i, h in enumerate(headers):
            #print(i)
            #print(h)
            print('| %%%ds ' % item_widths[i] % h, end='')
            txt = txt + ('| %%%ds ' % item_widths[i] % h)
                   
        print('|')
        txt = txt + '|\n'
        print(sep_line)
        
        txt = txt + sep_line +'\n'

    for row in values:
        for i, v in enumerate(row):
            print('| %%%ds ' % item_widths[i] % v, end='')
            txt = txt + ('| %%%ds ' % item_widths[i] % v)
        print('|')
        txt = txt + '|\n'
    print(sep_line)
    txt = txt + sep_line +'\n'
    #print(txt)
    return txt




def plot_sheep_conc_trend(txt, png_op, png_title):
    pep_li=('GAN-pep1','GAN-pep2','GAN-pep3','GAN-pep4','GAN-pep5','GAN-pep6',
              'GAN-pep7','GAN-pep8','Pep-1v','Pep-3m','Pep-6f','Pep-7c',
              'Pep-8m','Pep-9m_g','ACP_23_gag_24','AMP_23_gag_20','AFP_23_gag_18','AVP_23_gag1_23')
    
    prob_li=[]
    with open( txt , 'r') as f:
        #0_0.598  
        for l in f:        
            l=l.strip().split('_')
            prob_li.append( float(l[1]) )
  
    drug_value_d={}
    for idx,pep in enumerate(pep_li): 
        #print( pep, 7*idx, 7*(idx+1) )
        drug_value_d[pep] = prob_li[ 7*idx :7*(idx+1)]
        
    x_values = [1,2,4,8,16,32,64]

    nm=0
    for drug_name, prob_li in drug_value_d.items():
        #print(drug_name, prob_li)
        nm+=1
        if nm%3==0:
            plt.plot(x_values, prob_li, marker='*', linestyle='--', label=drug_name)
        elif nm%3==1:
            plt.plot(x_values, prob_li, marker='*', linestyle='-.', label=drug_name)
        else:
            plt.plot(x_values, prob_li, marker='^', linestyle=':', label=drug_name)

    plt.xlim(0, 90)
    #plt.ylim(0, 1)
    plt.xlabel("Concentration(uM)", fontsize=9)
    plt.ylabel("Probability", fontsize=9)
    plt.title("Tset_Set2(sheep blood) predicted probability", fontsize=10)
    #Show legend
    plt.legend(fontsize=7.4)
    plt.savefig(f'{png_op}/{png_title}.png')
    plt.close()





