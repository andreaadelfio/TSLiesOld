''' This module contains the implementation of the FOCuS algorithm for change point detection. '''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
try:
    from modules.plotter import Plotter
    from modules.utils import Data
    from modules.nn import get_feature_importance
except:
    from plotter import Plotter
    from utils import Data
    from nn import get_feature_importance


class Quadratic:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self):
        return f'Quadratic: {self.a}x^2+{self.b}x'

    def __sub__(self, other_quadratic):
        #subtraction: needed for quadratic differences
        return Quadratic(self.a-other_quadratic.a, self.b-other_quadratic.b)

    def __add__(self, other_quadratic):
        #addition: needed for quadratic differences
        return Quadratic(self.a+other_quadratic.a, self.b+other_quadratic.b)

    def evaluate(self, mu):
        return np.maximum(self.a*mu**2 + self.b*mu, 0)

    def update(self, X_T):
        return Quadratic(self.a - 1, self.b + 2*X_T)

    def ymax(self):
        return -self.b**2/(4*self.a) 

    def xmax(self):
        if (self.a==0)and(self.b==0):
            return 0
        else:
            return -self.b/(2*self.a)

    def dominates(self, other_quadratic):
        return (self.b>other_quadratic.b)and(self.xmax()>other_quadratic.xmax())


def focus_step(quadratic_list, X_T):
    new_quadratic_list = []
    global_max = 0
    time_offset = 0
    
    if not quadratic_list: #list is empty
        
        if X_T <= 0:
            return new_quadratic_list, global_max, time_offset
        else:
            updated_q = Quadratic(-1, 2*X_T)
            new_quadratic_list.append(updated_q)
            global_max = updated_q.ymax()
            time_offset = updated_q.a
            
    else: #list not empty: go through and prune
        
        updated_q = quadratic_list[0].update(X_T) #check leftmost quadratic separately
        if updated_q.b < 0: #our leftmost quadratic is negative i.e. we have no quadratics
            return new_quadratic_list, global_max, time_offset
        else:
            new_quadratic_list.append(updated_q)
            if updated_q.ymax() > global_max:   #we have a new candidate for global maximum
                global_max = updated_q.ymax()
                time_offset = updated_q.a

            for q in quadratic_list[1:]+[Quadratic(0, 0)]:#add on new quadratic to end of list
                updated_q = q.update(X_T)

                if new_quadratic_list[-1].dominates(updated_q):
                    break #quadratic q and all quadratics to the right of it are pruned out by q's left neighbour
                else:
                    new_quadratic_list.append(updated_q)

                    if updated_q.ymax() > global_max:   #we have a new candidate for global maximum
                        global_max = updated_q.ymax()
                        time_offset = updated_q.a
        
    return new_quadratic_list, global_max, time_offset

def plot_quadratics(quadratic_list, threshold, T):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("FOCuS step")
    
    ax.set_xlabel("$\mu$")
    ax.set_ylabel("$S_{T}(\mu)$", rotation=0)
    
    # ax.set_ylim(-1, threshold+1)
    # ax.set_xlim(-0.2, 5.2)
    
    
    ax.axhline(threshold, color='C1')

    mu = np.linspace(0, 5, 100) #the x-axis for the plot
    
    for q in quadratic_list:
        ax.plot(mu, q.evaluate(mu), label=f'$\\tau={q.a+T+1}$')
    
    ax.axhline(0, color='C0')

    if quadratic_list:
        ax.legend()
    return fig

def focus(X, threshold, plot=False):
    quadratic_list = []
    
    for T in range(len(X)):
        quadratic_list, global_max, time_offset = focus_step(quadratic_list, X[T])
        
        if plot:
            plot_quadratics(quadratic_list, threshold, T)
            plt.show()
        
        if global_max > threshold:
            return global_max, time_offset+T+1, T
        
    return 0, len(X)+1, len(X) #no change found by end of signal

def trigger(tiles_df, y_cols, y_pred_cols, threshold, model = None):
    """Run the trigger algorithm on the dataset.
    """
    if not os.path.exists('data/focus'):
        os.makedirs('data/focus')
    if not os.path.exists('data/focus/plots'):
        os.makedirs('data/focus/plots')

    bsize = 500
    anomalies_dict = []
    for key, key_pred in zip(y_cols, y_pred_cols):
        anomalies = 0
        print(f'{key}...', end=' ')
        signal = tiles_df[key] - tiles_df[key_pred]
        count = 0
        old_count, old_stopping_time = 0, 0
        while count+bsize < len(signal):
            sub_signal = signal[count:count+bsize].reset_index(drop=True)
            sigma = np.std(sub_signal)
            significance, changepoint, stopping_time = focus(sub_signal, threshold * sigma)
            if changepoint is not None and stopping_time < bsize and significance > 0:
                if count == old_count + old_stopping_time:
                    last_anomaly = anomalies_dict.pop()
                    new_count = last_anomaly[1]
                    new_changepoint = last_anomaly[2]
                    new_stopping_time = last_anomaly[3] + stopping_time
                    new_anomaly = (key, new_count, new_changepoint, new_stopping_time, str(tiles_df['datetime'][new_count+new_changepoint]), str(tiles_df['datetime'][count+stopping_time]), significance, sigma, threshold)
                else:
                    anomalies += 1
                    new_anomaly = (key, count, changepoint, stopping_time, str(tiles_df['datetime'][count+changepoint]), str(tiles_df['datetime'][count+stopping_time]), significance, sigma, threshold)
                anomalies_dict.append(new_anomaly)
                old_count, old_stopping_time = count, stopping_time
            count += stopping_time if stopping_time > 0 else bsize
        print(f'{anomalies} anomalies')

    support_vars = ['SUN_IS_OCCULTED', 'SOLAR']
    for key, count, changepoint, stopping_time, start_datetime, stop_datetime, significance, sigma, threshold in anomalies_dict:
        start = count+changepoint-100
        end = count+stopping_time+100
        signal = tiles_df[start:end]
        changepoint = count+changepoint
        stopping_time = count+stopping_time
        figs, axs = plt.subplots(6 + len(support_vars), 1, sharex=True, figsize=(18, 14), num=f'burst_{key}')
        plt.tight_layout()
        axs[0].set_title(f'Anomaly in {key}, with s = {round(significance / sigma):.2f} $\\sigma$ at $T = ${start_datetime}/{stop_datetime}')

        for i, (face, face_pred) in enumerate(zip(y_cols, y_pred_cols)):
            face_color = "black" if face != key else None
            axs[i].plot(signal['datetime'], signal[face], label=face, color=face_color)
            axs[i].plot(signal['datetime'], signal[face_pred], label=face_pred, color="red")
            axs[i].axvline(signal['datetime'][stopping_time], color='C1', lw=0.7)
            axs[i].axvline(signal['datetime'][changepoint], color='red', lw=0.7)
            axs[i].legend(loc="upper right")

        axs[1+i].plot(signal['datetime'], signal[key] - signal[f'{key}_pred'], label=f"{key} residual")
        axs[1+i].axhline(significance, color='C1', label="$\\sigma$")
        axs[1+i].fill((signal['datetime'][changepoint], signal['datetime'][stopping_time], signal['datetime'][stopping_time], signal['datetime'][changepoint]), (-5, -5, 15, 15), color="red", alpha=0.1, label="anomalous interval") 
        axs[1+i].axvline(signal['datetime'][stopping_time], color='C1', label="detection time $T$", lw=0.7)
        axs[1+i].axvline(signal['datetime'][changepoint], color='red', label="start point $\\tau$", lw=0.7)
        axs[1+i].set_ylim(min(signal[key] - signal[f'{key}_pred']), 1.01 * max(signal[key] - signal[f'{key}_pred']))
        axs[1+i].set_xlim(signal['datetime'][start], signal['datetime'].iloc[-1])
        axs[1+i].legend(loc="upper right")

        for j, var in enumerate(support_vars):
            axs[2+i+j].plot(signal['datetime'], signal[var], color="green", label=var)
            axs[2+i+j].legend(loc="upper right")
        
        axs[-1].set_xlabel("$datetime$")
        plt.tight_layout()
        figs.subplots_adjust(hspace=0)

        figs.savefig(f'data/focus/plots/{key}_{signal["datetime"][changepoint]}.png')
        plt.close(figs)
        if model:
            col_range = y_cols
            col_selected = [col for col in signal.columns if col not in y_cols + y_pred_cols + ['datetime', 'TIME_FROM_SAA', 'SUN_IS_OCCULTED', 'LIVETIME', 'MET', 'START', 'STOP', 'LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'SAA_EXIT', 'IN_SAA']]
            get_feature_importance(f"data/focus/plots/{key}_{signal['datetime'][changepoint]}_lime.png", inputs_outputs_df = signal[changepoint:stopping_time], col_range = col_range, col_selected = col_selected, model = model, show=False, num_sample=10)
            get_feature_importance(f"data/focus/plots/{key}_{signal['datetime'][stopping_time+50]}_lime.png", inputs_outputs_df = signal[changepoint+50:stopping_time+50], col_range = col_range, col_selected = col_selected, model = model, show=False, num_sample=10)

    focus_res = pd.DataFrame(anomalies_dict, columns=['face', 'start', 'changepoint', 'stopping_time', 'start_datetime', 'stop_datetime', 'significance', 'sigma', 'threshold'])
    focus_res.to_csv('data/focus/anomalies.csv', index=False)
    return anomalies_dict


if __name__ == '__main__':
    y_cols = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    y_pred_cols = [col + '_pred' for col in y_cols]

    fermi_data = pd.read_pickle('data/model_nn/1/pk/frg.pk')
    nn_pred = pd.read_pickle('data/model_nn/1/pk/bkg.pk')
    
    nn_pred = nn_pred.assign(**{col: nn_pred[cols_init] for col, cols_init in zip(y_pred_cols, y_cols)}).drop(columns=y_cols)
    tiles_df = Data.merge_dfs(nn_pred, fermi_data)
    Plotter(df=tiles_df, label='tiles').df_plot_tiles(x_col='datetime', marker=',',
                                                        show=True, smoothing_key='pred')

    trigger(tiles_df, y_cols, y_pred_cols, 1)