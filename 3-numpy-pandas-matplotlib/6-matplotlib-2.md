# Matplotlib and seaborn 2

## 1. Introduction
- last lesson univariate, now bivariate visualization
- used for looking at causation

## 2. Scatterplots and Correlation
- looking at two quantitative variables
- one var on x, other along y
- one point per observation in data
- correlation coefficient r
    - where 0 is weaker, 1 (or inverse -1) is stronger
    - only captures linear relationships, so can change to log
- use `plt.scatter` or `sb.regplot`
    - pass the x and y axes labels
    - `regplot` only works for linear

## 3. Overplotting, Transparency, and Jitter
- _overplotting_: a blob or too discrete, with too many points in a small area
- use sampling to plot a smaller number of points
- or plot points with transparency so overlaps get darker
- with too discrete (checkerboard kind of looking points), add jitter to spread points
- use `alpha`, `x_jitter`, `y_jitter`, `scatter_kws`

## 4. Heat Maps
- grid of cells, number of data points in each cell counted
- cell given different value based on count
    - color for emphasis
- alternative to adding jitter or transparency to large-data scatterplot
- tradeoffs with larger () or smaller data (noise)
- use `plt.hist2d` and add `bins`
    - calculate bins with `np.arange` for both x and y bin edges
- seaborn has `heatmap` function

## 5. Practice
- make a basic scatterplot and heatmap

## 6. Violin Plots
- quantitative against qualitative
- imagine for example plotting team scores per team
- compare it to jittered scatterplot to see clearly _categorical_ data
- convert classes to ordered categorical types: `pd.api.types.CategoricalDtype`
    - pass in your array of qualitative strings for the `categories`
- then use `sb.violinplot`
    - set color, `inner = None`, more

## 7. Box Plots
- another quantitative against qualitative
- box and whiskers
- outlier points beyond end of whiskers
- summary statistics: violin, explanatory: box plot "for simplicity and focus"
- as above but use `sb.boxplot`

## 8. Practice
- make a basic violin plot

## 9. Clustered Bar Charts
- use for qualitative vs. qualitative
- cluster bars together (like low-med-hi across three experiments)
- use `sb.countplot(data = df, x = cat_var_1, hue = cat_var_2)`
- or use heatmap instead: `sb.heatmap`
	- first turn things into matrix
	- then pass to the heatmap method

## 10. Practice
- practice categorical plots comparing fuel type per car class

## 11. Faceting
- facet by categorical variables to break down patterns into parts
	- think of violin plot as similar to faceted histogram
- keep axis scales and limits consistent across plots for subsets of data
- make plots per subset with `sb.FacetGrid`
	- `col` is for the faceted variable
	- then call the `.map` on the result to plot histograms
```Python
bin_edges = np.arange(-3, df['num_var'].max()+1/3, 1/3)
g = sb.FacetGrid(data = df, col = 'cat_var')
g.map(plt.hist, "num_var", bins = bin_edges)
```
- add even more variables like `col_wrap` when plotting many categories

## 12. Adaptation of Univariate Plots
- still comparing quantitative to quantitative
- use bar charts to plot means of second variable instead
	- include `y` arg in `sb.barplot`
	- plots uncertainty whiskers
	- change whiskers to show sd instead of avg: `ci = 'sd'`
	- use `pointplot` if no zero lower bound wanted for bars
	- turn off line by setting line to empty string
- can also adjust histograms to show something other than count through `weights`

## 13. Line Plots
- compare two quantitative variables
- emphasizes relative change
- avoids unnecessary zeros on y-axis
- inappropriate if nominal variable on x-axis
- often used over time ("time series plot"), with one measurement per time value
- use `plt.errorbar` passing in data, x and y will give wonky results
	- instead need to construct in multiple steps
	- make sure it's quantitative over quantitative
```Python
# set bin edges, compute centers
bin_size = 0.25
xbin_edges = np.arange(0.5, df['num_var1'].max()+bin_size, bin_size)
xbin_centers = (xbin_edges + bin_size/2)[:-1]

# compute statistics in each bin
data_xbins = pd.cut(df['num_var1'], xbin_edges, right = False, include_lowest = True)
y_means = df['num_var2'].groupby(data_xbins).mean()
y_sems = df['num_var2'].groupby(data_xbins).sem()

# plot the summarized data
plt.errorbar(x = xbin_centers, y = y_means, yerr = y_sems)
plt.xlabel('num_var1')
plt.ylabel('num_var2')
```
- can also plot histogram, scatterplot, [linspace](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html)

## 14. Practice
- practice plotting distribution and avg of quantitative over qualitative
- their solution for plotting avg fuel efficiency per manufacturer with 80+ cars:
```Python
# data setup
fuel_econ = pd.read_csv('./data/fuel_econ.csv')

THRESHOLD = 80
make_frequency = fuel_econ['make'].value_counts()
idx = np.sum(make_frequency > THRESHOLD)

most_makes = make_frequency.index[:idx]
fuel_econ_sub = fuel_econ.loc[fuel_econ['make'].isin(most_makes)]

make_means = fuel_econ_sub.groupby('make').mean()
comb_order = make_means.sort_values('comb', ascending = False).index

# plotting
base_color = sb.color_palette()[0]
sb.barplot(data = fuel_econ_sub, x = 'comb', y = 'make',
           color = base_color, order = comb_order, ci = 'sd')
plt.xlabel('Average Combined Fuel Eff. (mpg)')
```

## 15. Lesson Summary
- review of the plot types we met
- how to adapt univariate plots for bivariate data, too

## 16. Postscript
- also consider multivariate visualizations
- color for third variable (`color` or `c`) in scatterplot
- hue for FacetGrid
- color palettes through `sb.palplot(color_palette(n_colors=n))`
- faceting across two variables:
```Python
g = sb.FacetGrid(data = df, col = 'cat_var2', row = 'cat_var1', size = 2.5, margin_titles = True)
g.map(plt.scatter, 'num_var1', 'num_var2') 
```

## 17. Extra: Swarm Plots
- example of using `sb.swarmplot` vs violin or bivariate boxplot

## 18. Extra: Rug and Strip Plots
- `sb.stripplot` example
- `sb.rugplot` example joined with scatterplot

## 19. Extra: Stacked Plots
- examples of creating stacked bar charts
- reference: https://eagereyes.org/techniques/stacked-bars-are-the-worst
