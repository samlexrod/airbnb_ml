def get_word_cloud(target_columns):
    """
    To show a word cloud of targeted columns
    
    parameter
    ---------
    target_columns : a list of columns to show in the word cloud    
    """
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    target_columns = ' '.join(target_columns)

    wordcloud = WordCloud(width=480, height=480, max_font_size=20, min_font_size=10).generate(target_columns)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()

def apply_regression_models(dataframe, dependent_var, splits=3, n_estimators=100):
    """
    There are two models applying here:
        - Linear Regression
        - Random Forest Regression with 1000 n_estimators
    
    parameter
    ---------
    dataframe : the dataframe containing both the dependent and independent variables
    dependent_var : the target variable to predict
    splits : the number of split in the time series cross validation
    n_estimators : the number of trees in the random forest version of the regression
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
    from statsmodels.stats.stattools import durbin_watson
    import numpy as np
    import warnings
    
    # Prevent error of one split
    if splits < 2:
        warnings.warn("Splits were reset to 2. One split is not allowed.")
        splits = 2

    # Instantiating model and cv
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=n_estimators)
    tss = TimeSeriesSplit(n_splits=splits)
    
    # Resetting index for the cross validation
    dataframe.reset_index(drop=True, inplace=True)

    # Sorting time series
    dataframe = dataframe.sort_values('date')

    # Separating predictors and dependent variable
    X = dataframe.drop(dependent_var, axis=1)
    y = dataframe[[dependent_var]]
    
    score_dict = {}

    # Magic
    for test_num, split_index in enumerate(tss.split(X), 1):
        train_index, test_index = split_index 

        # Split descriptors
        min_train_idx, max_train_idx = train_index.min(), train_index.max()
        min_test_idx, max_test_idx = test_index.min(), test_index.max()
        train_pct = max_train_idx / X.shape[0]
        test_pct = (max_test_idx - min_test_idx) / X.shape[0]

        print(f"SPLIT {test_num}:", 
              f"TRAIN {min_train_idx} to {max_train_idx} or {train_pct:2.2%} |",
              f"TEST {min_test_idx} to {max_test_idx} or {test_pct:2.2%}")

        # Time series split
        X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit
        lr.fit(X_train, y_train)    
        rf.fit(X_train, y_train.values.ravel())    

        # Linear Predict
        y_hat_linear = lr.predict(X_test)       

        # Forest Predict
        y_hat_forest = rf.predict(X_test)

        def evaluate_model(model, model_name, y_pred):
            """
            Model evaluation metrics
            """
            score = model.score(X_test, y_test)
            score_dict.setdefault(model_name, [])
            score_dict[model_name].append(score)
            residuals = y_test - y_pred.reshape(-1, 1)
            MSE = mean_squared_error(y_test, y_pred)
            d_w = durbin_watson(residuals)[0] 

            print('-'*70)
            print(f"\tModel: {model_name}")
            print(f"\tCoefficient of Determinaion: {score:2.2%}",
                  f"\n\tMean Squared Errors: {MSE:2,.2f}",
                  f"\n\tDurbin-Watson: {d_w:2.1f}")            

        linear_model_name = 'Linear Regression'
        evaluate_model(lr, linear_model_name, y_hat_linear)

        forest_model_name = 'Random Forest Regressor'
        evaluate_model(rf, forest_model_name, y_hat_forest)

        print('_'*70)

    print(f"The linear average score is {np.mean(score_dict[linear_model_name]):2.2%}")
    print(f"The forest average score is {np.mean(score_dict[forest_model_name]):2.2%}")
    
    
    return {'LinearModel': lr, 'EnsembleModel': rf}

def vif(dataframe):
    """
    Creates a series of variance inflation factors to determine their multicollinearity.
    parameter
    ---------
    dataframe : the final reduced dataset with substantial explanatory features correlated with
    dependent feature.
    """
    import warnings
    from statsmodels.tools.tools import add_constant
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # ignore divide by zero warning
    warnings.filterwarnings("ignore")
    
    X = add_constant(dataframe)
    columns = X.columns
    
    # Factor container
    vif_factors = []
        
    shape_idx = X.shape[1]
    
    # For all features
    for i in range(shape_idx):
        print(f"[{'='*i}]{i/(shape_idx-1):2.2%}", flush=True, end='\r')
        
        vif_factor = variance_inflation_factor(X.values, i)
            
        # Factor for the feature
        vif_factors.append(vif_factor)
              
    warnings.filterwarnings("default")
        
    return pd.Series(vif_factors, index=columns)

def get_top_collinearity(dataframe, show_top10=False):
    """
    To extract only top collinearity.
    Note: the date feature must be named 'date'.
    Note: the price variable must be named 'price'
    
    parameter
    ---------
    dataframe : the final dataframe to pass to the models with no objects
    """
    collinearity_price = dataframe.corr()[['price']].drop('price', axis=0).apply(abs)
              
    if show_top10:
        display(collinearity_price.sort_values('price', ascending=False)[:10])
    
    # Keeping only 5% and above    
    reduced_features = collinearity_price.sort_values(
        'price', ascending=False).where(lambda x: x > .05).dropna().index.tolist()
    
    return dataframe[['price'] + ['date'] + reduced_features]

def iqr_outlier_detect(series):
    """
    To identify outliers using the quartile approach
    
    parameter
    ---------
    series : the series of values to check for in a pandas series object
    """
    import matplotlib.pyplot as plt
    q1 = series.quantile(.25) 
    q3 = series.quantile(.75)
    q3_q1 = q3 - q1
    upper_wisker = q3 + q3_q1 * 1.5
    lower_wisker = q1 - q3_q1 * 1.5
    
    print(f"{series.describe()}")
    
    series.plot.box(vert=False, figsize=(10, 1))
    plt.show()
    
    return lower_wisker, upper_wisker

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
            zip_ref.extractall('data')
            zip_ref.close()
        except:
            print(f'{file} already extracted!')
        
    return os.listdir('data')

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
        
        # Plot
        self.listings[['row_null_pct']].plot.hist(
            alpha=.5, 
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
        all by feature name, one at a time

        parameters
        ----------
        feature_name : the name of the feature to transform   
        """
        
        try:
            series = self.dataframe[feature_name]
  
            self.feature_name = feature_name

            # Awesomely splitting and getting unique content
            join_func = lambda x: ''.join(x)
            re_cond = '[a-zA-Z,]' # Char only and comma
            re_sub = '[^a-zA-Z,]' # No char or comma
            re_sub_func = lambda x: re.sub(re_sub, '', x) # subtitute non char or commas

            # Find characters and commas to split
            content = series.str.findall(re_cond).apply(
                join_func).str.split(',').tolist()
            
            unique_content_list = []
            for sublist in content:
                for item in sublist:
                    unique_content_list.append(item)
            
            # Avoid duplicates
            unique_content_list = list(dict.fromkeys(unique_content_list))
            
            # Do not inlcude any null, blanks, etc.
            unique_content_list = [val for val in unique_content_list if val not in ('', 'None', 'nan', 'NaN')]
            
            # Reset container
            self.dummy_frame = pd.DataFrame()
            
            # Finding content
            for val in unique_content_list:
                
                # Series name
                name = f"{feature_name}_{val}"
                
                # Find unique content and flag as 1 if found
                dummy = series.apply(re_sub_func).str.find(
                    val).apply(lambda x: 0 if x < 0 else 1).rename(name)

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