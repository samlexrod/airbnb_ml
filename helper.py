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
        Merges the dataframes for analysis
        """
        self.df_merged = self.calendar.merge(
            self.listings,
            left_on='listing_id',
            right_on='id',
            how='left',
            suffixes=['', '_listing'])

    def correlation_status(self, column_list, show_values=True):
        """
        Plotting the correlations to keep track of multicollinearity and corrlation with price.
        
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
        
    def listing_row_null_dist(self, bins, title):
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
        
def split_create_dummy(series):
    """
    To split features separated by delimiter as listings
    
    parameters
    ----------
    series : the data as a series object    
    """
    
    # Empty frame
    split_dummy = pd.DataFrame()

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

    # Finding content
    for val in unique_content:
        dummy = series.apply(re_sub_func).str.find(
            val).apply(lambda x: 0 if x < 0 else 1)

        # Creating the dummy frame
        split_dummy = pd.concat([split_dummy, dummy.to_frame(val)], axis=1)

    return split_dummy
        
        

