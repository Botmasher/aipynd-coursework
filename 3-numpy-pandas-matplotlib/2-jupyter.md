# Lesson 2: Jupyter Notebooks

## 1. Instructor
- same as last lesson

## 2. What are Jupyter notebooks?
- use data, code, images, math, other stuff in your browser
- text cells can contain markdown text or code
- render code output in notebook instead of separate window
- example of Knuth's _literate programming_ where docs are alongside code
- connects user <-> browser <-> notebook server <-> (kernel, notebookfile)
    - kernel doesn't have to run on Python
    - server can run anywhere
    - set up on your server or somewhere like Amazon EC2

## 3. Installing Jupyter Notebook
- run `conda install jupyter notebook`
- also available through pip

## 4. Launching the notebook server
- start a server: `jupyter notebook`
- note that directory you open the server is where files will be saved
- navigate to the port to connect to the server
- browse tabs and create a new notebook
- install Notebook Conda to manage environments: `conda install nb_conda`
- shut down notebook by clicking checkmark on server home and "shutdown"
    - unsaved changes are lost and code must be rerun
- shut down server with <kbd>ctrl + c</kbd>

## 5. Notebook interface
- the green box in a new notebook is a cell
- "command palette" search to see commands available, like to merge cells
- autosave but save manually with <kbd>escape</kbd> then <kbd>s</kbd>
- "file" contains ways to export

## 6. Code cells
- write code and execute in the cell
- code executed in one cell is available to other cells

## 7. Markdown cells
- (basic markdown for text cells)
- math using LaTeX symbols

## 8. Keyboard shortcuts
- get faster by using the keybindings

## 9. Magic keywords
- commands starting with one or two percent signs
- run commands in cells to control notebook or even system
    - example: set matplotlib to work interactively: `%matplotlib`
    - example: time a function in your code using `%timeit my_func()`
    - example: time a whole cell using `%%timeit` at the top of the cell
- these are restricted to the normal Python kernel
- use embedded visualizations: `%matplotlib inline`
    - for high-res screens: `%config InlineBackend.figure_format = 'retina'`
- magic command for interactive debugger: `%pdb`
    - quit it by entering <kbd>q</kbd>
- there are many more commands, so see links for this lesson

## 10. Converting notebooks
- notebooks are JSON files ending in `.ipynb`
- convert to another format: `jupyter nbconvert --to html notebook.ipynb`

## 11. Creating a slideshow
- toggle on a slideshow presentation under `View`
- choose how each cell shows up in dropdown menu
    - slides move through left to right
    - subslides come in with up and down
    - fragments appear on button press
    - skip skips, notes leaves as speaker notes
- create slideshow: `jupyter nbconvert notebook.ipynb --to slides`
- convert and display: `jupyter nbconvert notebook.ipynb --to sildes --post serve`

## 12. Finishing up
- Anaconda plus Jupyter augments "productivity and general well-being"
