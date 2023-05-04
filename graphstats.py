import pandas as pd
import numpy as np
import pickle

##### Locations they been to in past week - and how frequent it is
##### Spread stats for last week at said locations
##### Age, sex
##### Household number
##### Raw duration of time spent with infected neighbors
##### One hot encoding of types of activities participated in or weighted with how long they did those
#####
# Week states

def flatten(lst):
    flat_lst = []
    for elem in lst:
        if isinstance(elem, list):
            flat_lst.extend(flatten(elem))
        else:
            flat_lst.append(elem)
    return flat_lst

class GraphStats:
    def __init__(self, va_activity_loc_assign, va_activity_locations, va_disease_outcome_target, va_disease_outcome_training, va_household, va_person, va_population_network, va_residence_locations) -> None:
        self.va_activity_loc_assign = va_activity_loc_assign
        self.va_activity_locations = va_activity_locations
        self.va_disease_outcome_target = va_disease_outcome_target
        self.va_disease_outcome_training = va_disease_outcome_training
        self.va_household = va_household
        self.va_person = va_person
        self.va_population_network = va_population_network
        self.va_residence_locations = va_residence_locations
        self.infected_pids = set(va_disease_outcome_training.query('state == "I"')['pid'].unique())


    ##gets age and sex of a specified person
    def get_age_sex(self, pid):
        res = self.va_person.query('pid == @pid').iloc[0,:]
        return (res['age'],res['sex'])
    #gets the week up to and including that day ex: 6 = [0,6]
    def get_week_states(self, pid, day):
        sts = self.va_disease_outcome_training.query('pid == @pid and day >= @day - 6 and day <= @day')['state'].to_numpy()
        mapping = {"S": 0, "I": 1, "R": 2}
        map_func = np.vectorize(mapping.get)
        return map_func(sts)

    ### gets household members including person
    def household_members(self, pid):
        hid = self.va_person.query('pid == @pid')['hid'].squeeze()
        return self.va_person.query('hid == @hid')['pid'].to_numpy()

        
    def get_household_size(self, pid):
        hid = self.va_person.query('pid == @pid')['hid'].squeeze()
        return self.va_household.query('hid == @hid')['hh_size'].squeeze()
    
    # return the pids of people that a person come into contact with
    def get_contacted_people(self, pid):
        interactions = self.va_population_network.query('pid1 == @pid or pid2 == @pid')
        pid1 = interactions['pid1'].to_numpy()
        pid2 = interactions['pid2'].to_numpy()
        pids = np.concatenate((pid1, pid2))
        pids = pids[pids!=pid]
        return pids
    
    # get the number of infected people that this person come into contact with, run get_contacted_people first to get pids
    def get_num_contact_with_infected(self, pid, pids):
        num_contact_with_infected = sum(1 for pid in pids if pid in self.infected_pids)
        return num_contact_with_infected
        
    # how long a person is in contact with a infected people on a given day
    def get_raw_time_with_infected_day(self, pid, day, interactions=None,pids = None, disease_info=None):
        time = 0
        if(interactions is None):
            pids = self.get_contacted_people(pid)
            disease_info = self.va_disease_outcome_training.query('pid in @pids and state == "I"')

        disease = disease_info.query("day == @day")
        #print("info: ", disease_info)
       # print(interactions)
        for neighbor in disease.iterrows():
            n_pid = neighbor[1]['pid']
            
            time += interactions.query('pid1 == @n_pid or pid2 == @n_pid')['duration'].sum()

        return time
    
    def get_raw_time_with_infected_week(self, pid, day):
        pids = self.get_contacted_people(pid)
        times = []
        disease_info = self.va_disease_outcome_training.query('pid in @pids and state== "I"')
        num_contact_with_infected = sum(1 for pid in pids if pid in self.infected_pids) 
        num_contact = pids.size
        for i in range(day-6, day+1):
            times.append(self.get_raw_time_with_infected_day(pid, i, interactions, pids, disease_info))
    
        return np.array(times), num_contact_with_infected, num_contact
    
    def get_activity_vector(self, pid):
        user_activities = self.va_activity_loc_assign.query('pid == @pid')
        activity_vector = np.zeros(6)
        for i in range(1, 7):
            activity_vector[i-1] += user_activities.query('activity_type == @i')['duration'].sum()
        return activity_vector

    def get_location_durations(self, pid):
        location_durations = {}
        user_activities = self.va_activity_loc_assign.query('pid == @pid')
        for index, row in user_activities.iterrows():
            location_durations[row['lid']] = location_durations.get(row['lid'], 0) + row['duration']
        return location_durations
    
    def get_dataset(self):
        dataset = []
        pid_list = self.va_person["pid"].unique()
        for pid in pid_list:
            print(pid)
            age, sex = self.get_age_sex(pid) # get age and sex
            household_size = self.get_household_size(pid)    
            activity_vector = list(self.get_activity_vector(pid).astype(int))
            # if there's no residence set residence_duration = 0
            # sum up activity durations
            location_durations = self.get_location_durations(pid)
            residence_duration = 0
            activity_duration = 0
            for location, duration in location_durations.items(): 
                if location > 1000000: # residence
                    residence_duration += duration
                else:
                    activity_duration += duration
            for day in range(6, 57):
                infected_week_time = list(self.get_raw_time_with_infected_week(pid, day=day))
                row = [pid, day, age, sex, household_size, activity_vector, residence_duration, activity_duration, infected_week_time]
                flattened_row = flatten(row)
                # print(f"\t\t{flattened_row}")
                dataset.append(flattened_row)
        return dataset

def main():
    va_activity_loc_assign = pd.read_csv('va_activity_location_assignment.csv.gz', compression='gzip').iloc[:,1:]
    va_activity_locations = pd.read_csv('va_activity_locations.csv.gz', compression='gzip').iloc[:,1:]
    va_disease_outcome_target = pd.read_csv('va_disease_outcome_target.csv.gz', compression='gzip').iloc[:,1:]
    va_disease_outcome_training = pd.read_csv('va_disease_outcome_training.csv.gz', compression='gzip')
    va_household = pd.read_csv('va_household.csv.gz', compression = 'gzip').iloc[:,1:]
    va_person = pd.read_csv('va_person.csv.gz', compression='gzip')
    va_population_network = pd.read_csv('va_population_network.csv.gz', compression='gzip')
    va_residence_locations = pd.read_csv('va_residence_locations.csv.gz', compression='gzip').iloc[:,1:]
    test = GraphStats(va_activity_loc_assign, va_activity_locations, va_disease_outcome_target, va_disease_outcome_training, va_household, va_person, va_population_network, va_residence_locations)  
    # dataset = test.get_dataset()
    pid = 4545426
    contacted_pids = test.get_contacted_people(pid)
    num_contact = contacted_pids.size
    num_contact_with_infected = test.get_num_contact_with_infected(pid, contacted_pids)
    print(num_contact)
    print(num_contact_with_infected)
    # with open('final_dataset.pkl', 'wb') as f:
    #     pickle.dump(dataset, f)


if __name__ == "__main__":
    main()