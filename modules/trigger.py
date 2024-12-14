'''
This module contains the implementation of the FOCuS algorithm for change point detection.
'''
import os
import json
from math import log
import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
try:
    from modules.plotter import Plotter
    from modules.utils import Data, Logger, logger_decorator
    from modules.config import TRIGGER_FOLDER_NAME, PLOT_TRIGGER_FOLDER_NAME
except:
    from plotter import Plotter
    from utils import Data, Logger, logger_decorator
    from config import TRIGGER_FOLDER_NAME, PLOT_TRIGGER_FOLDER_NAME



class Curve:
    '''
    From the original python implementation of
    FOCuS Poisson by Kester Ward (2021). All rights reserved.
    '''

    def __init__(self, k_T, lambda_1, t=0):
        self.a = k_T
        self.b = -lambda_1
        self.t = t

    def __repr__(self):
        return "({:d}, {:.2f}, {:d})".format(self.a, self.b, self.t)

    def evaluate(self, mu):
        return max(self.a * log(mu) + self.b * (mu - 1), 0)

    def update(self, k_T, lambda_1):
        return Curve(self.a + k_T, -self.b + lambda_1, self.t - 1)

    def ymax(self):
        return self.evaluate(self.xmax())

    def xmax(self):
        return -self.a / self.b

    def is_negative(self):
        # returns true if slope at mu=1 is negative (i.e. no evidence for positive change)
        return (self.a + self.b) <= 0

    def dominates(self, other_curve):
        return (self.a + self.b >= other_curve.a + other_curve.b) and (self.a * other_curve.b <= other_curve.a * self.b)

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

class Trigger:
    logger = Logger('Trigger').get_logger()

    def focus_step_quad(self, quadratic_list, X_T):
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


    def focus_step_curve(self, curve_list, k_T, lambda_1):
        '''
        From the original python implementation of
        FOCuS Poisson by Kester Ward (2021). All rights reserved.
        '''
        if not curve_list:  # list is empty
            if k_T <= lambda_1:
                return [], 0., 0
            else:
                updated_c = Curve(k_T, lambda_1, t=-1)
                return [updated_c], updated_c.ymax(), updated_c.t

        else:  # list not empty: go through and prune

            updated_c = curve_list[0].update(k_T, lambda_1)  # check leftmost quadratic separately
            if updated_c.is_negative():  # our leftmost quadratic is negative i.e. we have no quadratics
                return [], 0., 0,
            else:
                new_curve_list = [updated_c]
                global_max = updated_c.ymax()
                time_offset = updated_c.t

                for c in curve_list[1:] + [Curve(0, 0)]:  # add on new quadratic to end of list
                    updated_c = c.update(k_T, lambda_1)
                    if new_curve_list[-1].dominates(updated_c):
                        break
                    else:
                        new_curve_list.append(updated_c)
                        ymax = updated_c.ymax()
                        if ymax > global_max:  # we have a new candidate for global maximum
                            global_max = ymax
                            time_offset = updated_c.t

        return new_curve_list, global_max, time_offset

    def trigger_face(self, signal, face, diff):
        '''
        From the original python implementation of
        FOCuS Poisson by Kester Ward (2021). All rights reserved.
        '''
        result = {f'{face}_std': [], f'{face}_offset': [], f'{face}_significance': []}
        curve_list = []
        std = np.std(signal)
        for T in tqdm(signal.index, desc=face):
            # start_window, end_window = max(0, T-120), min(len(signal), T+120)
            # std = np.std(signal[start_window:end_window])
            x_t = signal[T]
            if diff[T] > 60:
                curve_list = []
            curve_list, global_max, offset = self.focus_step_quad(curve_list, x_t)
            result[f'{face}_std'].append(std)
            result[f'{face}_offset'].append(offset)
            result[f'{face}_significance'].append(np.sqrt(2 * global_max))
        return result

    @logger_decorator(logger)
    def trigger(self, tiles_df, y_cols, y_pred_cols, thresholds: dict):
        '''Run the trigger algorithm on the dataset.

        Args:
            `tiles_df` (pd.DataFrame): dataframe containing the data
            `y_cols` (list): list of columns to be used for the trigger
            `y_pred_cols` (list): list of columns containing the predictions
            `thresholds` (dict): thresholds dictionary for each signal

        Returns:
            dict: dict containing the anomalies
        '''
        if not os.path.exists(TRIGGER_FOLDER_NAME):
            os.makedirs(TRIGGER_FOLDER_NAME)
        if not os.path.exists(PLOT_TRIGGER_FOLDER_NAME):
            os.makedirs(PLOT_TRIGGER_FOLDER_NAME)

        triggs_dict = {}
        diff = tiles_df['MET'].diff()
        pool = multiprocessing.Pool()
        results = []
        for face, face_pred in zip(y_cols, y_pred_cols):
            result = pool.apply_async(self.trigger_face, (tiles_df[face] - tiles_df[face_pred], face, diff))
            results.append(result)

        for result in results:
            triggs_dict.update(result.get())
        pool.close()
        pool.join()

        triggs_df = pd.DataFrame(triggs_dict)
        triggs_df['datetime'] = tiles_df['datetime']
        return_df = triggs_df.copy()
        # triggs_df.to_csv(os.path.join(PLOT_TRIGGER_FOLDER_NAME, 'triggers.csv'), index=False)
        mask = False
        for face in y_cols:
            mask |= triggs_df[f'{face}_significance'] > thresholds[face] * triggs_df[f'{face}_std']
        triggs_df = triggs_df[mask]

        count = 0
        anomalies_faces = {face: [] for face in y_cols}
        old_stopping_time = {face: -1 for face in y_cols}

        for index, row in tqdm(triggs_df.iterrows(), total=len(triggs_df), desc='Identifying triggers'):
            for face in y_cols:
                if row[f'{face}_significance'] > thresholds[face] * row[f'{face}_std']:
                    changepoint, stopping_time = row[f'{face}_offset']+index+1, index
                    significance = row[f'{face}_significance']
                    sigma_val = row[f'{face}_std']
                    datetime = str(row['datetime'])

                    if index == old_stopping_time[face] + 1 or changepoint <= old_stopping_time[face] + 60 and anomalies_faces[face]:
                        last_anomaly = anomalies_faces[face].pop()
                        new_changepoint = last_anomaly[1]
                        new_significance = last_anomaly[7]
                        datetime = last_anomaly[5]
                        max_significance = max(new_significance, significance)
                        max_point = index if significance > new_significance else last_anomaly[-1]
                        new_stopping_time = stopping_time
                        new_anomaly = (face, new_changepoint, new_changepoint, new_stopping_time, datetime, datetime, significance, max_significance, sigma_val, thresholds[face], max_point)
                    else:
                        count += 1
                        new_anomaly = (face, changepoint, changepoint, stopping_time, datetime, datetime, significance, significance, sigma_val, thresholds[face], changepoint)

                    anomalies_faces[face].append(new_anomaly)
                    old_stopping_time[face] = stopping_time
        
        anomalies_list = []
        for face in y_cols:
            anomalies_list += anomalies_faces[face]

        print('Merging triggers...', end=' ')
        merged_anomalies = {}
        for face, start, changepoint, stopping_time, start_datetime, stop_datetime, significance, max_significance, sigma_val, threshold, max_point in anomalies_list:
            if returned := self.is_mergeable(start, merged_anomalies, minutes=1):
                start, old_start = returned
                if start < old_start:
                    merged_anomalies[start] = merged_anomalies[old_start]
                    del merged_anomalies[old_start]
                elif start > old_start:
                    start = old_start
                merged_anomalies[start][face] = {'changepoint': changepoint, 'stopping_time': stopping_time, 'start_datetime': start_datetime, 'stop_datetime': stop_datetime, 'significance': significance, 'max_significance': max_significance, 'sigma_val': sigma_val, 'threshold': threshold, 'max_point': max_point}
            else:
                merged_anomalies[start] = {face: {'changepoint': changepoint, 'stopping_time': stopping_time, 'start_datetime': start_datetime, 'stop_datetime': stop_datetime, 'significance': significance, 'max_significance': max_significance, 'sigma_val': sigma_val, 'threshold': threshold, 'max_point': max_point}}
        print(f'{len(merged_anomalies)} anomalies in total.')

        with open(os.path.join(PLOT_TRIGGER_FOLDER_NAME, 'detections.csv'), 'w') as f:
            f.write("start_datetime,start_met,triggered_faces\n")
            for start, anomaly in sorted(merged_anomalies.items(), key=lambda x: int(x[0]), reverse=True):
                f.write(f"{tiles_df['datetime'][int(start)]},{tiles_df['MET'][int(start)]},{'_'.join(anomaly.keys())}\n")
        return merged_anomalies, return_df
            
    def is_mergeable(self, start: int, merged_anomalies: dict, minutes = 1) -> tuple[int, int]:
        '''Check if the current anomaly is mergeable with any of the previous anomalies.

        Args:
            `start` (int): start time of the current anomaly
            `merged_anomalies` (dict): dict of anomalies

        Returns:
            tuple: (start, anomaly_start) if the anomaly is mergeable, False otherwise
        '''
        for anomaly_start in merged_anomalies:
            if start - 60 * minutes < anomaly_start < start + 60 * minutes:
                return start, anomaly_start
        return False


if __name__ == '__main__':
    y_cols = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    y_pred_cols = [col + '_pred' for col in y_cols]

    fermi_data = pd.read_pickle('data/background_prediction/1/pk/frg.pk')
    nn_pred = pd.read_pickle('data/background_prediction/1/pk/bkg.pk')
    
    nn_pred = nn_pred.assign(**{col: nn_pred[cols_init] for col, cols_init in zip(y_pred_cols, y_cols)}).drop(columns=y_cols)
    tiles_df = Data.merge_dfs(nn_pred, fermi_data)
    Plotter(df=tiles_df, label='tiles').df_plot_tiles(x_col='datetime', marker=',',
                                                        show=True, smoothing_face='pred')

    Trigger().trigger(tiles_df, y_cols, y_pred_cols, 3)
