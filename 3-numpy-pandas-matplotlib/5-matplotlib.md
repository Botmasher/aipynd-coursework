# Matplotlib and Seaborn

## 1. Instructor
- visualize your data

## 2. Introduction
- _univariate_ (single-variable) visualizations
- keep goals in mind and use variables that are key in investigation

## 3. Tidy Data
- tidy dataset
    - each variable is a column
    - each observation is a row
    - each type of observational unit is a table
- examples of observational units: patient, treatment
- so a table combining patient names and treatments is not tidy
- not the only way; consider bivariate plotting matrix for heatmap

## 4. Bar Charts
- qualitative variable, where each level has its own bar
- order sorted nominal data with frequency
- but do not do this for ordinal data, like agree-to-disagree
- horizontal orientation for space
- as well as numpy and panda, include `matplotlib.pyplot` and `seaborn`
    - pyplot convention: `plt`
- use `sb.countplot` to create chart
    - pass `data` and `x` label (or `y` for horizontal)
    - `order` to sort chart, using pandas series `value_counts` to sort elements
    - set `color`, use `.color_palette` to get current color

## 5. Absolute vs. Relative Frequency
- absolute for count of data in each category
- relative for proportion of data falling in each category
- change to relative frequency to find proportion relative to whole
- use `np.arange`, change `plt.xtics`, `.ytics`
- label as proportion with `.ylabel`
```Python
# get correct sort order (here based on a 'type' quality)
sort_order = pkmn_types['type'].value_counts().index

# calculate proportions
totals = pkmn_types['species'].unique().shape[0]
highest_total = pkmn_types['type'].value_counts()[0]
proportion = highest_total / totals

# axis tickmarks for proportions
ticks = np.arange(0, max_prop, 0.02)
tick_labels = ['{:0.2f}'.format(v) for v in ticks]

# plot and label data
sb.countplot(data=pkmn_types, y="type", order=sort_order, color="blue")
plt.xticks(ticks * totals, tick_labels)
plt.xlabel("proportion")
```

## 6. Counting Missing Data
- create pandas table of missing data with `.isna().sum()`
- use seaborn barplot to visualize: `sb.barplot(counts_var.index.values, counts_var)`
- (`countplot` avoids having to do extra summarization work required by `barplot`)

## 7. Bar Chart Practice
- bar chart with absolute then one with sorted relative frequency

## 8. Pie Charts
- relative frequencies for qualitative variables
- donut plots are similar
- bar charts are preferred
- show how whole breaks down into parts and have few slices
- plot from 12 o'clock then slice from there
- use `plt.pie`, passing in `wedgeprops` for donut

## 9. Histograms
- 

## 10. Histogram Practice
- 

## 11. Figures, Axes, and Subplots
- 

## 12. Choosing a Plot for Discrete Data
- 

## 13. Descriptive Statistics, Outliers and Axis Limits
- 
