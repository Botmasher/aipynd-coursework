# Lesson 4: Pandas

## 1. Instructors
- same, introducing data manipulation and analysis with pandas

## 2. Introduction
- short for _panel data_
    - work with data series and data frames
    - work with labeled and relational data
- included with Anaconda
- check version: `conda list pandas`
- install it: `conda install pandas=0.22`

## 3. Why use pandas?
- large datasets need to be checked
    - missing/bad values
    - outliers
- do basic data analysis
- label rows and cols
- calculate rolling stats on time series data
- handle NaN
- load different formats
- join/merge datasets

## 4. Creating pandas Series
- 1d array-like objects
- include: `import pandas as pd`
- initialize: `pd.Series(data=[], index=[])`
- assign label and index to each element in the series
- index in first col, data in second
- get `.shape`, `.ndim` (dimensionality), `.size` of data
- read `.index` and `.values` separately
- check index labels using `in` command

## 5. Accessing and Deleting Elements in pandas Series
- access or modify the series elements
    - use the index labels in square brackets
    - provide list of index labels in square brackets
    - give standard numerical indexes in square brackets
- `.loc` ensures you're using defined index
- `.iloc` ensures you're passing a numerical index
- reassign using dict key-like modification syntax
- modifications happen in the original series

## 6. Arithmetic Operations on pandas Series
- do element-wise operations between series and numbers
- modify the data by adding, subtracting, multiplying, dividing all values
- calculate `np.sqrt` or `np.exp` or `np.power` of each element
- can operate on select indexes: `series_name.loc(['index1', 'index2']) * 2`
- operation must be defined for all data types it's applied to
    - this is true for mixed types like numbers and strings 

## 7. Quiz
- create a series, operate on elements, use boolean indexing to search for elements

## 8. Creating pandas DataFrames
- 2d array-like object with rows, cols
- imagine creating one from a dictionary containing many series
    - row names are keys, series are values
    - like an items series per customer
- initialize: `pd.DataFrame(my_dict)`
    - displayed in table like spreadsheet
    - row labels built from union of series dictionary values
    - columns based on dictionary keys
- NaN values when no data for a cell
- if index labels not present for series, numerical index used starting with `0`
- again check `.size`, `.shape`, `.ndim`
- use `DataFrame(my_dict, index=[], columns=[])` for specific rows and cols
    - all arrays in dictionary must have same length

## 9. Accessing Elements in pandas DataFrames
- access rows, cols, elements using row and col labels
- get a specific col: `my_df[['col_name']]`
- get a specific row: `my_df.loc(['row_name'])`
- get a specific row and col: `my_df['col_name']['row_name']`
- add a col: `my_df['new_col_name'] = [row_values, ...]`
- add new items to a new dataframe then `append` that col to an existing dataframe
- restrict additions to specific rows: `my_df['col'][i:]`
- insert column in a specific location: `my_df.insert(loc, 'name', [values, ...])`
- delete with `pop` (cols) and `drop` (rows if `axis=0`, cols when `axis=1`)
- rename with `rename` passing it a dictionary remapping old to new labels
    - pass the dict to `index` param for rows
    - pass this to the `columns` parameter for cols
- use `.set_index` to change the value of the row index
    - instead of using row labels use the data from one of the columns (why?)

## 10. Dealing with NaN
- clean your data, detecting and fixing errors
- most common type of bad data is missing numbers
- find your NaNs: `my_df.isnull().sum().sum()`
    - get a boolean DataFrame containing all pandas-falsy values
    - this finds the NaNs
    - then sum up the number of NaNs per col
    - then finally sum up all cols
- find your non-NaNs: `my_df.count()`
- remove or replace missing values
    - remove from rows: `my_df.dropna(axis=0, inplace=False)`
    - remove from cols: `my_df.dropna(axis=1, inplace=False)`
    - pass `inplace=True` to change the DataFrame itself
    - replace with zeros: `my_df.fillna(0)`
    - forward fill with previous row/col vals: `my_df.fillna(method='ffil', axis=0)`
    - forward filling only works when there are previous values to use!
- interpolate instead from values along an axis
    - consider methods like linear or backfill

## 11. Quiz
- create a dict from existing series
- make DataFrame with dict
- replace NaNs with col means
    - this was just `.fillna(my_df.mean(), inplace=True)`

## 12. Loading Data into a pandas DataFrame
- use dbs from different formats, including CSV
- use `pd.read_csv()`
- take a look at the first/last rows instead of all: `my_df.head()`, `.tail()`
    - pass an integer to get n rows instead of 5
- get descriptive stats with `.describe()` on df (or single col in brackets)
- get `.max`, `.min`, `.corr`, ...
- use `.group_by` to get info
    - imagine summing all salaries per year: `my_df.group_by(['Year'])['Salary'].sum()`
    - do the same but get the `.mean` for average per year
    - group by name instead to get totals paid per person
    - group year, then department, then sum salaries: `...(['Year', 'Department'])...`

## 13. Getting Set Up for the Mini-Project
- this is for doing it locally instead of the project workspace

## 14. Mini-Project
- read CSVs of prices for three tech stocks with `pd.read_csv`
    - use `Date` as the index column
    - set `parse_dates=True`
    - choose columns to use
- create date range for comparing stocks with `pd.date_range`
- use date range as index for new DataFrame comparing stocks
- rename join columns of individual stocks so they have different names
- use `.join` to them to the new DataFrame
- find and remove any NaN rows
- calculate the mean, median, standard deviation, correlation
- plot the data
