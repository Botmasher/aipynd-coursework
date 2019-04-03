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
- quantitative variable, value ranges
- bars include values on the left end up to those on the right end
- `plt.hist` much like bar chart
- set the `bin` value
- consider that `arange` does not include the max value
- pass `sb.distplot` panda series with all data
    - default `bins` count larger than matplotlib
    - defaults to include density curve
    - if turning the curve off, why aren't you just using matplotlib?

## 10. Histogram Practice
- create a simple histogram

## 11. Figures, Axes, and Subplots
- instead of `hist` could've also set up a `plt.figure()`
    - add axes to it: `.add_axes`
    - run the histogram on the axes: `.hist()`
- could also do this with `countplot` to get bar chart
- `subplot` to split visuals of plots into rows and columns
    - you can, say, have multiple separate plots side by side
    - not doing this might overlay plots

## 12. Choosing a Plot for Discrete Data
- plot discrete data as a bar chart
- plot discrete data as a histogram but taking less `rwidth` to separate bars
- do not plot ordinal data as a histogram 

## 13. Descriptive Statistics, Outliers, and Anomalies
- these plots go beyond descriptive statistics
- plots tell skewednesss, unimodal, multimodal
- see errors in data
- pay attention to unusual values
- histogram: are there outliers?
- use `plt.xlim` for skewed data to chop of edges (give upper and lower bounds)

## 15. Scales and Transformations
- also change scaling from linear
- finance often deals in different scales
    - capture small diffs on low end, large at high end
    - multiplicative rather than arithmetic diffs
    - make tickmarks *2 of previous rather than +2
- skewedness again, but with an axis transform
- use `plt.xscale('log')`
    - modify the bin boundaries using some logarithmicky math

## 16. Practice
- make histograms both cutting and scaling the axis

## 17. Lesson Summary
- learned about plot types
- learned techniques for dealing with data

## 18. Extra: Kernel Density Estimation
- plot a KDE using `sb.distplot`
- notes about how this works and how to interpret
