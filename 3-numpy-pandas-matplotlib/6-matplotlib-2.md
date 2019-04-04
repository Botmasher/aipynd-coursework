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
- 

## 10. Practice
- 

## 11. Faceting
- 

## 12. Adaptation of Univariate Plots
- 

## 13. Line Plots
- 

## 14. Practice
- 

## 15. Lesson Summary
- 

## 16. Postscript
- 

## 17. Extra: Swarm Plots
- 

## 18. Extra: Rug and Strip Plots
- 

## 19. Extra: Stacked Plots
- 
