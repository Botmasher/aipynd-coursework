# Lesson 1: Anaconda

## 1. Instructor
- former physicist, neuroscientist, data scientist
- Anaconda as OS for working with big data, simplifying package management

## 2. Introduction
- get Python versions to work with each other
- Anaconda libraries built for data science
- core is `conda` package manager and environment manager
- use `conda create -n your_environment_name python=3`
- once in the environment, see name, prompt and check out installed packages
- install packages for data: `conda install numpy pandas matplotlib`
- install Jupyter notebooks for working with code
- packages are separated out so no conflicting with other versions

## 3. What is Anaconda?
- repeating what's said above
- Anaconda comes with a bunch of useful preinstalled data science packages
- Miniconda comes with just `conda` and Python
- reduce issues and conflicts from libraries
- like pip except that it's for data science and not exclusive to Python
- Anaconda packages are precompiled, lag behind releases thanks to contributors, stable
- virtual environment manager for separating out package installations
    - example: can't install multiple numpys easily otherwise
- export `requirements.txt` list and load dependencies

## 4. Installing Anaconda
- install as instructed in previous lesson, see your install with `conda list`
- upgrade with `conda upgrade conda` then `conda upgrade -all`

## 5. Managing packages
- just install with `conda install package_name1 package_name2 ...`
- specify package version like `conda install numpy=1.10`
- dependencies automatically get installed (like `scipy` depends on `numpy`)
- remove with `conda remove package_name`
- update with `conda update package_name`
- search with `conda search *searchterm*`
- so then, how do you think you would install 

## 6. Managing environments
- create your own environment: `conda create -n environment_name package_name1 ...`
    - the `-n` sets the name
    - the name is followed by a list of packages to include in the environment
- specify the Python version during the create like `python=3` at the end
- activate a created environment: `conda activate environment_name`
- check out the prompt to see your environment name
- install packages beyond the basics and whatever you installed
    - any packages installed within the environment are only available there
- leave the environment: `source deactivate`

## 7. More environment actions
- save all environment packages to YAML: `conda env export > environment.yaml`
- create from YAML: `conda env create -f environment.yaml`
- see your environment names: `conda env list`
- the environment used when you aren't within one of your named ones is `root`
    - in my Anaconda it seems to be called `base`
- remove an environment: `conda env remove -n environment_name`

## 8. Best practices
- it may be helpful to have separate environments for 2 vs 3
- create environments for each new project, even non-data heavy Flask ones
- sharing code on GitHub, create environment file and including in repo
    - include a pip `requirements.txt` with `pip freeze` as well for non-Anaconda
- read conda dev blogs for more

## 9. On Python versions at Udacity
- Python3, Jupyter, Python2 getting retired
- `print` statement as most common source of errors
