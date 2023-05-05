import numpy as np
import pandas as pd
import pickle
from graphstats import GraphStats

def flatten(lst):
    flat_lst = []
    for elem in lst:
        if isinstance(elem, list):
            flat_lst.extend(flatten(elem))
        else:
            flat_lst.append(elem)
    return flat_lst

class PreProcessor:
    def __init__(self, va_activity_loc_assign, va_activity_locations, va_disease_outcome_target, va_disease_outcome_training, va_household, va_person, va_population_network, va_residence_locations) -> None:
        self.va_activity_loc_assign = va_activity_loc_assign
        self.va_activity_locations = va_activity_locations
        self.va_disease_outcome_target = va_disease_outcome_target
        self.va_disease_outcome_training = va_disease_outcome_training
        self.va_household = va_household
        self.va_person = va_person
        self.va_population_network = va_population_network
        self.va_residence_locations = va_residence_locations
        self.graphstats = GraphStats(va_activity_loc_assign, va_activity_locations, va_disease_outcome_target, va_disease_outcome_training, va_household, va_person, va_population_network, va_residence_locations)
    # gets the training labels associated with a specific person should be 50
    def get_training_labels(self, pid):
        outcomes = self.va_disease_outcome_training.query("pid == @pid")
        labs = {}
        for i in range(6, 57):
            lab = 0
            outs = outcomes.query('day >= @i - 6 and day <= @i')
            for row in outs.iterrows():
                if(row[1]['state'] == 'I'):
                    lab = 1
                    break
            labs[i] = lab
        return labs
    def get_training_labels_all(self):
        ret = {}
        for pid in pd.unique(self.va_person['pid']):
           ret[pid] = self.get_training_labels(pid)
        return ret
    def pickle_processing(self):
        data = []
        with open('dataset.pkl', 'rb') as f:
            data = pickle.load(f)
        d = []
        with open('dataset_day56.pkl','rb') as f:
            d = pickle.load(f)
        labels = []
        with open('train_labels.pkl','rb') as f:
            labels = pickle.load(f)
        filtered_data = [list(filter(lambda x: x <= 4500000, row)) for row in data]
        datas = np.array(data)
        pids = datas[:,0]
        days = datas[:,1]
        filtered_data = np.array(filtered_data)[:,1:]
        d = np.array(d)
        pids_56 = d[:,0]
        days_56 = d[:,1]
        d = d[:,2:]
        print(d)
        print(filtered_data)

        pids_total = np.concatenate((pids,pids_56))
        days_total = np.concatenate((days, days_56))
        data_total = np.concatenate((filtered_data, d), axis=0)
        print(pids_total)
        print(days_total)
        print(data_total)
        return data_total, pids_total, days_total, labels
    
    def filtered_data_process(self):
        dataset = []
        pid_list = pd.unique(self.va_disease_outcome_training['pid'])
        labs = {}
        for pid in pid_list:
            age, sex = self.graphstats.get_age_sex(pid)
            activity_vector = list(self.graphstats.get_activity_vector(pid).astype(int))
            location_durations = self.graphstats.get_location_durations(pid)
            residence_duration = 0
            activity_duration = 0
            for location, duration in location_durations.items(): 
                if location > 1000000: # residence
                    residence_duration += duration
                else:
                    activity_duration += duration
            for day in pd.unique(self.va_disease_outcome_training.query('pid == @pid')['day']):
                infected_week_time = list(self.graphstats.get_raw_time_with_infected_week(pid, day=day-1))
                row =  [pid, day-1, age, sex, activity_vector, residence_duration, activity_duration, infected_week_time]
                flattened_row = flatten(row)
                dataset.append(flattened_row)
                if(pid not in labs):
                    labs[pid] = {}
                labs[pid][day-1] = 1
        dataset = np.array(dataset)
        pids = dataset[:,0]
        days = dataset[:,1]
        dataset = dataset[:,2:]

        print(labs)
        return dataset, pids, days, labs

        
