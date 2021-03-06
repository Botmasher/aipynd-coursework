# Lesson 3: NumPy

## 1. Instructors
- collaboration between data scientist and computational physicist

## 2. Introduction to NumPy
- learn about numerical Python and dealing with _ndarrays_
- included with Anaconda
- `conda list numpy` checks version (`!conda list numpy` in Jupyter)
- update version with `conda install numpy=1.13`

## 3. Why Use NumPy?
- core is n-dimensional array
    - grid that can take many shapes
    - all elements must be of the same type
    - enables quick computations
- include it: `import numpy as np` (this is the standard abbreviation)
- try getting the mean of 100 million random numbers
    - in standard Python this can take say 9 secs
    - in NumPy it may take say 0.09 secs
- ML problems often involve ndarrays, like holding pixel data
- many packages are built on top of NumPy

## 4. Creating and Saving NumPy ndarrays
- create an array using notation or with functions
- 1D array: `x = np.array([1, 2, 3, 4])`
- read the datatype: `x.dtype`
- the array shape: `x.shape` => tuple with number of rows, columns
- the array size: `x.size` => total number of elements in array
- the _rank_  is the number of dimensions, like the rank-1 array above
- note that mixed ints and strings will create a unicode string list!
- ints and floats mixed will upcast everything
- save array into a file: `np.save('array_name', x)`
- load file: `np.save('array_name.npy')`

## 5. Using Built-in Functions to Create ndarrays
- generate ndarrays from nothing: `x = np.zeros((3,4))`
    - defaults to floats
    - can also do `.ones` or a `.full((rows, cols), n)` passing any constant
- _identity matrix_ square matrix with ones along main axis and zero everywhere else
    - try it: `np.eye(5)` will generate five-row matrix with diagonal ones
    - use other values on the diagonal: `np.diag([10, 20, 30, ...])`
- create range of evenly spaced integers: `np.arange(start, stop, step)`
    - stop is required and is exclusive
    - be careful about non-integer steps
- float steps: `np.linspace(start, stop, count, endpoint)`
    - count how many numbers to chop along the line
    - boolean `endpoint` allows excluding the stop like `arange` does
- create rank-2 arrays with ranges and `.reshape`
    - pay attention to division
    - like you can convert 20-element rank-1 to 4x5 rank-2 array
    - but you cannot convert 20-element rank-1 array to a 5x5 array
    - reshape an array: `np.arange(20).reshape((10, 2))`
- create random matrix: `np.random.random((3, 3))`
    - defaults to 0 inclusive, 1 exclusive
    - use this to initialize weights for a neural network
- create random integer matrix: `np.random.randint(lower, upper, (rows, cols))`
- random arrays from distributions: `np.random.normal(avg, sd, size=(rows, cols))`
    - check out the `.mean` and `.sd` to see how close they ended up

## 6. Quiz
- practice generating and reshaping an array

## 7. Accessing, Deleting, and Inserting
- ndarrays are mutable
- use square brackets to specify index
- access rank-2 elements with two comma-separated numbers between square brackets
- delete elements: `np.delete(ndarray, index_list, axis)`
    - `axis=0` for rows, `1` for columns
- insert elements: `np.insert(ndarray, index, elements, axis)`
- stack arrays
    - vertically: `np.vstack((top_array, bottom_array))`
    - horizontally: `np.hstack((left_array, right_array.reshape(rows, cols)))`

## 8. Slicing ndarrays
- slice with colons
- specify slice for each array dimension
- imagine grabbing bottommost, rightmost 3x3 from a 5x5 matrix
    - try it this way: `ndarray[1:4, 2:5]`
    - or leave off ending: `ndarray[1:, 2:]`
- grab all rows in one column as a rank-1 array: `ndarray[:, 2]`
- grab all rows in one column as rank-2 array: `ndarray[:, 2:3]`
- assigning slices to new variables does _not_ copy slice into that variable
    - we call it a "view" of the original array
    - any changes within that view change those elements in the original
- create a copy instead
    - from NumPy: `x = np.copy(ndarray[1:, 2:])`
    - on the array: `x = ndarray[1:, 2:].copy()`
- grab elements one diagonal above a diagonal: `np.diag(ndarray, k=1)`
    - `k` defaults to `0` for main diagonal
- grab only uniques: `np.unique(ndarray)`

## 9. Boolean Indexing, Set Operations, and Sorting
- get the index of integers less than a value: `print(x[(x > 10) & (x < 20)]`
    - this is called _boolean indexing_
- compare sets for common elements
    - use `np.intersect1d`, `np.setdiff1d`, `np.union1d`
- sort rank 1 and rank 2 arrays
    - using `sort` as a function sorts out of place, but method does in place
    - syntax: `np.sort(x)`
    - combine with `.unique` to get only uniques
    - use keyword `axis` to sort by rows or cols: `np.sort(x, axis=0)`

## 10. Quiz
- slice an ndarray looking for odds-only using boolean indexing

## 11. Arithmetic Operations and Broadcasting
- use symbols or functions to get the same output: `+` or `np.add`
- same goes for `divide`, `multiply`, `subtract`
- _broadcasting_ involves arrays of different shapes
    - operators operate on same-shape or broadcastable arrays
- calculate `np.exp` and `np.sqrt` of each element
- raise members to a `pwr`
- all the statistical methods like `min`, `.mean` and `.sd`
- modify each number without complicated loops
    - add a number to each member: `n + ndarray_name`
    - behind the scenes NumPy is broadcasting an array for `n` to make the shape
- add a different number to each col: `rank_1_name + rank_2_ndarray_name`
    - simplest case has as many members in rank 1 array as there are cols in the other
- add a different number to each row: `rank_2_name + rank_2_ndarray_name`
    - simplest case has single-number subarray rows added to each row in the other
- see the documentation for broadcasting workings and rules

## 12. Quiz
- practice creating a rank-2 array and adding different numbers to each column

## 13. Mini-Project Setup
- project requires Anaconda and can be completed in workspace or notebooks locally

## 14. Mini-Project
- data normalization also known as _feature scaling_ for similar range of values
    - normalize data to fall between 0 and 1
- _mean normalization_ distributes values in interval around zero
    - values evenly distributed around `0`
    - guarantees the average is `0`
- import NumPy and create a rank 2 array of random integers between 0 and 5,000
    - array will have 1000 rows and 20 columns
- normalize the data using the mean normalization equation
    - for each column subtract col mean and divide the result by col std
- check that the mean is roughly zero and the avg min and max fall on either side
- now split the dataset for ML into:
    - a _training set_ (60%)
    - a _cross validation set_ (20%)
    - a _test set_ (20%)
    - these should be picked at random
- create rank 1 array with random permutation of row indices from normalized matrix
    - `np.random.permutation(n)` creates random permutation of ints from `0` to `n-1`
    - note you're making random _indexes_ here not yet grabbing values
    - pass normalized array shape's row count to permutation function
- now use those random row indexes to create the datasets
    - you can just use fancy indexing here: `array_name[random_indexes]`
    - that worked even with a rank 2 array when permutations was a rank 1!
