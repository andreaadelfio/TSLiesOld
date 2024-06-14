''' This module contains the implementation of the FOCuS algorithm for change point detection. '''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
    from scripts.plotter import Plotter
    from scripts.utils import Data
except:
    from plotter import Plotter
    from utils import Data


class Curve:
    '''
    From https://github.com/kesward/FOCuS
    '''
    def __init__(self, k_T, lambda_1, t=0):
        self.a = k_T
        self.b = -lambda_1
        self.t = t
        ## a log(mu) + b(mu-1)
        ## t contains time offset, because lambda_1 is incorporated into b.
    
    def __repr__(self):
        return f'Curve: {self.a} log(mu) + {self.b} (mu-1), t={self.t}.'

    def evaluate(self, mu):
        return np.maximum(self.a*np.log(mu) + self.b*(mu-1), 0)

    def update(self, k_T, lambda_1):
        return Curve(self.a + k_T, -self.b + lambda_1, self.t-1)

    def ymax(self):
        return self.evaluate(self.xmax())

    def xmax(self):
        return -self.a/self.b

    def is_negative(self):
        #returns true if slope at mu=1 is negative (i.e. no evidence for positive change)
        return (self.a + self.b) < 0

    def dominates(self, other_curve):
        self_root = -self.a/self.b #other non mu=1 root: the curve's "length"
        other_root = -other_curve.a/other_curve.b
        self_slope = self.a + self.b  #slope at mu=1: the curve's "height"
        other_slope = other_curve.a + other_curve.b
        return (self_root > other_root)and(self_slope > other_slope)

def focus_step(curve_list, k_T, lambda_1):
    '''
    From https://github.com/kesward/FOCuS
    '''
    new_curve_list = []
    global_max = 0
    time_offset = 0

    if not curve_list: #list is empty
        if k_T <= lambda_1:
            return new_curve_list, global_max, time_offset
        else:
            updated_c = Curve(k_T, lambda_1, t=-1)
            new_curve_list.append(updated_c)
            global_max = updated_c.ymax()
            time_offset = updated_c.t
    else: #list not empty: go through and prune
        updated_c = curve_list[0].update(k_T, lambda_1) #check leftmost quadratic separately
        if updated_c.is_negative(): #our leftmost quadratic is negative i.e. we have no quadratics
            return new_curve_list, global_max, time_offset
        else:
            new_curve_list.append(updated_c)
            if updated_c.ymax() > global_max:   #we have a new candidate for global maximum
                global_max = updated_c.ymax()
                time_offset = updated_c.t

            for c in curve_list[1:]+[Curve(0, 0)]:#add on new quadratic to end of list
                updated_c = c.update(k_T, lambda_1)

                if new_curve_list[-1].dominates(updated_c):
                    break #quadratic q and all quadratics to the right of it are pruned out by q's left neighbour
                elif updated_c.is_negative():
                    pass #delete q and move on
                else:
                    new_curve_list.append(updated_c)

                    if updated_c.ymax() > global_max:   #we have a new candidate for global maximum
                        global_max = updated_c.ymax()
                        time_offset = updated_c.t
        
    return new_curve_list, global_max, time_offset

def plot_curves(curve_list, threshold=25, T=0):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"FOCuS step at detection time $T={T}$")
    
    ax.set_xlabel("$\mu$")
    ax.set_ylabel("$S_{T}(\mu)$", rotation=0)
    
    ax.set_ylim(-1, threshold+1)
    # ax.set_xlim(-0.2, 5.2)
    
    
    ax.axhline(threshold, color='C1')

    mu = np.linspace(0.2, 10, 100) #the x-axis for the plot
    
    for c in curve_list:
        ax.plot(mu, c.evaluate(mu), label=f'$\\tau={c.t+T+1}$')
    
    ax.axhline(0, color='C0')

    if curve_list:
        ax.legend(loc='upper left')
    return fig

def focus(X, lambda_1, threshold, plot=False):
    if np.ndim(lambda_1)==0:#scalar
        lambda_1 = np.full(X.shape, lambda_1)

    signif = []
    offsets = []

    curve_list = []
    for T in range(len(X)):
        curve_list, global_max, time_offset = focus_step(curve_list, X[T], lambda_1[T])
        signif.append(global_max)
        offsets.append(time_offset)
        
        if plot:
            plot_curves(curve_list, threshold, T)
            plt.show()
        
        if global_max > threshold:
            plot_curves(curve_list, threshold, T)
            plt.show()
            return global_max, time_offset+T+1, T, signif, offsets
        
    return 0, len(X)+1, len(X), signif, offsets #no change found by end of signal

def trigger(tiles_df, y_cols, y_pred_cols, threshold):
    """Run the trigger algorithm on the dataset.
    """
    dct_res = {}
    dct_offset = {}
    for key, key_pred in zip(y_cols, y_pred_cols):
        print("Focus trigger... Elaborating key: ", key)
        global_max, time_offset, instant, signif, offsets = focus(tiles_df[key], tiles_df[key_pred], threshold)
        anomaly_detected = np.argmax(np.array(signif))
        anomaly_start = anomaly_detected + offsets[anomaly_detected]
        if global_max == 0:
            start = global_max
            end = instant
        else:
            start = anomaly_start
            end = anomaly_detected + anomaly_start + 1

        print(start, end)
        figs, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        axs[0].plot(tiles_df['datetime'][start:end], np.sqrt(np.array(signif)*2), color="C0", label=f"detector {key}")
        axs[0].axhline(threshold, color="red")
        axs[0].set_ylabel(f'sigma level {key}')

        axs[0].set_title("Significance")
        

        axs[1].step(tiles_df['datetime'][start:end], tiles_df[key][start:end], where='pre', label=key, color="black")
        axs[1].step(tiles_df['datetime'][start:end], tiles_df[key_pred][start:end], where='pre', label=key, color="red")
        axs[1].set_ylabel(f'detector {key}')
        axs[1].axvline(tiles_df['datetime'][anomaly_start], color="red")
        axs[1].axvline(tiles_df['datetime'][anomaly_detected], color="red")

        figs.supxlabel('datetime')
        
        figs.savefig('plots/burst.png', dpi = 150)
        plt.show()

        dct_res[key] = signif
        dct_offset[key] = offsets

    focus_res = pd.DataFrame(dct_res)
    focus_offset = pd.DataFrame(dct_offset)
    focus_res.to_csv('data/focus' + '/trig.csv',
                     index=False, float_format='%.2f')
    focus_offset.to_csv('data/focus' + '/offset.csv',
                     index=False, float_format='%.2f')
    print("Done.")
    return focus_res


if __name__ == '__main__':
    y_cols = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    y_pred_cols = [col + '_pred' for col in y_cols]

    # Load dataset of foreground and background
    fermi_data = pd.read_pickle('data/model_nn/1/pk/frg.pk')
    nn_pred = pd.read_pickle('data/model_nn/1/pk/bkg.pk')
    
    nn_pred = nn_pred.assign(**{col: nn_pred[cols_init] for col, cols_init in zip(y_pred_cols, y_cols)}).drop(columns=y_cols)
    tiles_df = Data.merge_dfs(nn_pred, fermi_data)
    Plotter(df=tiles_df, label='tiles').df_plot_tiles(x_col='datetime', marker=',',
                                                        show=True, smoothing_key='pred')

    trigger(tiles_df, y_cols, y_pred_cols, 1)