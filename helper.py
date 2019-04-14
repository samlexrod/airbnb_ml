class Tracker:
    
    def __init__(self, trackers):
        """
        Instantiate this tracker with a list of list you want to track.
        
        parameters:
        -----------
        tracker : list of lists to keep track of features -required
        """
        import pandas as pd
        
        # Use pandas in the whole class
        global pd
        
        # Initializing trackers        
        self.tracker_names = ['all_assessed', 'predictors', 'engineered', 'possible', 
                         'dummies', 'irrelevant']
        
        # Assigning ordered trackers
        self.trackers = []     
        
        for name in self.tracker_names:
            self.trackers.append(trackers.get(name))
    
    def replace(self, tracker_name, value, new_value):
        """
        To replace misspelled words, etc.
        
        parameters:
        -----------
        tracker_name : the name of the tracker where the value will be replaced
        value : the value to be replaced
        new_value : the value replacing the original value
        
        Note: all fields are required
        """
        
        # Get the idx of tracker
        tracker_idx = self.tracker_names.index(tracker_name)
        
        # Target the location of value
        value_idx = self.trackers[tracker_idx].index(value)
        
        # Replace the value with new value
        self.trackers[tracker_idx][value_idx] = new_value
        
        
    def add(self, track_type, features, assessed=False):
        """
        To keep track of features

        parameters:
        ----------
        track_type : the specific tracker for the feature or features -required
            -predictors
            -possible
            -dummies
            -irrelevant
            -engineered
        features : list of features to add to tracker -required
        assessed : to add to all_assessed tracker when the feature or features is/are analyzed -default False
        """
        # Assign trackers
        all_assessed, predictors, engineered, possible, dummies, irrelevant = self.trackers
        
        # Pass features to trackers and clean duplicates
        def pass_and_clean(feature_holder):
            """"
            This function prevents duplicates
            """
            [feature_holder.append(x) for x in features if x not in feature_holder]
            return feature_holder

        if track_type == 'predictors':
            predictors = pass_and_clean(predictors)
        elif track_type == 'dummies':
            dummies = pass_and_clean(dummies)
        elif track_type == 'possible':
            possible = pass_and_clean(possible)
        elif track_type == 'irrelevant':
            irrelevant = pass_and_clean(irrelevant)
        elif track_type == 'engineered':
            engineered = pass_and_clean(engineered)
        else:
            print("Nothing added :(")

        # Add if the feature was analyzed
        if assessed:
            all_assessed = pass_and_clean(all_assessed)
        
    def check(self):
        """
        It returns a dataframe of the tracked features.
        
        It is usefull for filtering features by calling track.check().tracker_name
        where tracker_name is the name of the tracker.
            -predictors
            -possible
            -dummies
            -irrelevant
            -engineered
            -all_assessed
        """
        
        all_assessed, predictors, engineered, possible, dummies, irrelevant = self.trackers
        
        # All trackers
        tracker_list = [all_assessed, predictors, engineered, possible, 
                        dummies, irrelevant]

        # Empty dataframe
        df = pd.DataFrame()

        # Create dataframe of trackers by columns
        for tracker, name in list(zip(tracker_list, self.tracker_names)):

            # Create tracker columns df
            df_col = pd.DataFrame(tracker, columns=[name])

            # Combine trackers in df
            df = pd.concat([df, df_col], axis=1)

        return df
    
def unzip_files(zip_filename_list, folder=None):
    """
    Unzipping the datasets
    """
    from zipfile import ZipFile
    import os
    
    folder_names = []
    for file in zip_filename_list:
        
        # paths
        filename_path = os.path.join('data', file)
        folder_name = filename_path[:-4].replace("-", "_")
        
        # extracting
        try:
            zip_ref = ZipFile(filename_path)
            zip_ref.extractall(folder_name)
            zip_ref.close()
        except:
            print('File already extracted!')
        
        # appending folder names
        if filename_path.find('.') >= 0:
            folder_names.append(folder_name)
        
    return folder_names

def read_concat(folder_names, target_file_name):
    """
    Reading files of different folders with same naming convention
    """
    import os
    import pandas as pd
    
    df_con = pd.DataFrame()
    for folder in folder_names:
        target = os.path.join(folder, target_file_name)
        df = pd.read_csv(target)
        df['rowsource'] = target
        
        df_con = pd.concat([df_con, df], axis=0, sort=True).reset_index(drop=True)
    return df_con

def cleaning_dollar(series):
    """
    To clean dollar signs and commas
    """
    series = series.str.replace('$', '')
    series = series.str.replace(',', '').astype(float)
    
    return series

