import numpy as np
import pandas as pd

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
    # gets the training labels associated with a specific person should be 50
    def get_training_labels(self, pid):
        outcomes = self.va_disease_outcome_training.query("pid == @pid")
        labs = []
        for i in range(6, 56):
            lab = 0
            outs = outcomes.query('day >= @i - 6 and day <= @i')
            for row in outs.iterrows():
                if(row[1]['state'] == 'I'):
                    lab = 1
                    break
            labs.append(lab)
        return np.array(labs)