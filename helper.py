def unzip_files(zip_filename_list):
    """
    Unzipping the datasets
    
    parameters
    ----------
    zip_filename_list : the list of file names to unzip under the data folder
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
    # Function to handle nulls
    join_func = lambda x: ''.join(x) if type(x)!=float else x
    
    series = series.str.findall('[^$,]').apply(join_func).astype(float)
    
    return series

def cleaning_percent(series):
    """
    To clean percentages and convert to ratio format
    """    
    # Function to handle nulls
    join_func = lambda x: ''.join(x) if type(x)!=float else x
    
    series = series.str.findall('[^%]').apply(join_func).astype(float)/100
    
    return series

class AnalysisStatus:
    
    def __init__(self, calendar, listings):
        """
        Instantiating the calendar dataset
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        global sns
        global plt
        
        # Attributes
        self.calendar = calendar
        self.listings = listings
        
        self._merge()
        
    def _merge(self):
        """
        Merges the dataframes for correlation analysis with price
        
        parameters
        ----------
        updted_frame : updated dataframe to use for the correlation analysis
        """        
        self.df_merged = self.calendar.merge(
            self.listings,
            left_on='listing_id',
            right_on='id',
            how='left',
            suffixes=['', '_listing'])
        
    def colliniearity_table(self, include_perfect=True, multi_r=1):
        """
        Prints a table with respective feature-price collinearity and
        explanatory variable multicollinearity.

        parameters
        ----------
        multi_r : the above or equal treshold of multicollinearity to show in table
        exclude_perfect : to exclude perfect multicollinearity from the table
        """
        
        print("Processing collinearity table. Please wait a minute.")

        # Direction indicator function
        direction_func = lambda x: (x.abs_multi_r < 0).map({True: '-', False: '+'})

        # Absolute multicollinearity function
        abs_multi_r_func = lambda x: x.abs_multi_r.apply(abs)

        # Mutlicollinearity treshold function
        multi_r_func = lambda x: x.abs_multi_r >= multi_r

        # Exclude perfect muticollinearity function
        if include_perfect:
            perfect_func = lambda x: x.abs_multi_r >= 0
        else:
            perfect_func = lambda x: x.abs_multi_r != 1

        return self.df_merged.corr().reset_index().melt(
            id_vars=['index', 'price'],
            var_name='relation_to',
            value_name='abs_multi_r').rename(columns={
                'index': 'feature',
                'price': 'price_r'
            }).assign(
                direction=direction_func,
                abs_multi_r=abs_multi_r_func).where(perfect_func).where(
                    multi_r_func).dropna().query("feature!=relation_to")

    def correlation_heatmap(self, column_list, show_values=True):
        """
        Plotting the correlations to keep track of multicollinearity and correlation with price.
        
        parameters:
        -----------
        column_list : the list of columns to see on the heatmap
            use price_listing for listings price
        """
        col_interest = []
        
        # Avoid duplicates
        [col_interest.append(x) for x in ['price'] + column_list if x not in col_interest]
        
        # Merge price and listings
        price_corr = self.df_merged[col_interest]
        
        # Drop ids and set all as floats
        price_corr = price_corr.query("price==price").astype(float).corr()

        plt.figure(figsize=(14, 10))

        sns.heatmap(price_corr, annot=show_values, cmap='magma');
        
    def null_row_listingdist(self, bins, title):
        """
        Plotting the distribution of null values by state
        """
        
        # Adding row null counts to listings
        t_columns = self.listings.shape[1]
        h_columns = self.listings.isnull().sum(axis=1)
        self.listings.loc[:, 'row_null_pct'] = h_columns/t_columns
        
        # Makes sure state has consisntent cases
        upper_func = lambda x: x.state.str.upper()
        
        # Plot
        self.listings.assign(state=upper_func)[['state', 'row_null_pct']]\
            .reset_index()\
            .pivot(index='index',
                  columns='state',
                  values='row_null_pct')\
            .plot.hist(alpha=.5, 
                       bins=bins,
                       figsize=(12, 5),
                       title=title); 
        
    def null_row_feature_status(self, percentages=False, threshold=.10):
        """
        Getting values of null values of listings
        
        parameters
        ----------
        percentages : to see the results as a percentage of listings --default True
        threshold : to exclude anything less or equal to the passed value --default .10
        """
        output = self.listings.isnull().sum().where(lambda x: x > 0).dropna(
            ).sort_values(ascending=False)
        
        if percentages:
            return (output/self.listings.shape[0]).where(lambda x: x>threshold).dropna()
        else:
            return output.where(lambda x: x>threshold).dropna()
        
    def scatter_status(self, by):
        """
        Plotting the relationship between date and price by another category as color
        """
        
        # Plotting scaterplot
        plt.title(f'Prices Vs Dates by {by}')
        sns.scatterplot(x='date_num', 
                        y='price', 
                        hue=by, 
                        data=self.df_merged)
        
class DummySplit:
    
    def __init__(self, dataframe):
        """
        To split, encode, and add to dataset
        """
        import warnings
        import pandas as pd
        import re
        
        global warnings, pd, re
        
        # Attributes
        self.dummy_frame = pd.DataFrame()
        self.dataframe = dataframe
        self.dummy_columns = None
        self.columns = dataframe.columns.tolist()
        self.feature_name = None
        
    def split_create_dummy(self, feature_name):
        """
        To split features separated by delimiter as listings

        parameters
        ----------
        feature_name : the name of the feature to transform   
        """
        
        try:
            series = self.dataframe[feature_name]
  
            self.feature_name = feature_name

            # Awesomely splitting and getting unique content
            join_func = lambda x: ''.join(x)
            re_cond = '[a-zA-Z,]'
            re_sub = '[^a-zA-Z,]'
            re_sub_func = lambda x: re.sub(re_sub, '', x)


            content = series.str.findall(re_cond).apply(
                join_func).str.split(',').tolist()

            unique_content = []
            for sublist in content:
                for item in sublist:
                    unique_content.append(item)

            unique_content = list(dict.fromkeys(unique_content))

            unique_content = [val for val in unique_content if val not in ('', 'None', 'nan', 'NaN')]
            
            # Reset container
            self.dummy_frame = pd.DataFrame()
            
            # Finding content
            for val in unique_content:
                dummy = series.apply(re_sub_func).str.find(
                    val).apply(lambda x: 0 if x < 0 else 1)

                # Creating the dummy frame
                self.dummy_frame = pd.concat([self.dummy_frame, dummy.to_frame(val)], axis=1)  

            # Store dummy colums for dataframe additions
            self.dummy_columns = self.dummy_frame.columns.tolist()[:]
        
        except:
            warnings.warn("Invalid feature was passed... pass a feature in the dataframe")
            print(self.columns)            

    def add_dummies(self, drop_original=False):
        """
        To add created dummies to the dataset passed at the same location in memory

        parameters
        ----------
        dataframe : the dataframe to add the indicator variables into    
        """
        # To match split dumy index
        #self.dataframe.reset_index(inplace=True, drop=True)
        
        # Drop derived features
        try:
            self.dataframe.drop(self.dummy_columns, axis=1, inplace=True)
            print("Resetting Dummies :)")
        except:
            pass # First time try
        
        # Add new encoded features        
        for dummy in self.dummy_columns:
            self.dataframe.loc[:, dummy] = self.dummy_frame[dummy]
            
        # Remove original feature
        try:
            self.dataframe.drop(self.feature_name, axis=1, inplace=True)
        except:
            pass # Already removed