# Lesson 6: Scripting

## 1. Introduction
- environment, setup and scripting

## 2. Python Installation
- making sure you have a terminal and python installed

## 3. Install Python Using Anaconda
- install anaconda for data science stuff

## 4. Git Bash for Windows
- windows installation, not applicable to me

## 5. Running a Python Script
- use `python3` in the command line to run a script

## 6. Programming Environment Setup
- choose and use an editor

## 7. Editing a Python Script
- run a py script and copy-paste the output to show it ran

## 8. Scripting with Raw Input
- taking input using `input()`

## 9-10 Quiz & Solution
- example of taking input, converting to lists, formatting a message iterating through

## 11. Errors and Exceptions
- syntax errors vs run-time exceptions
- Python's various built-in exceptions like `NameError`

## 12. Quiz
- understand when syntax errors vs run-time errors possibly occur
- different error types

## 13. Handling Errors
- using `try`, `except`, `else`, `finally` to flow through explicit runtime exceptions

## 14-15 Practice & Solution
- use `try`-`except` to handle `ZeroDivisionError`

## 16. Accessing Error Messages
- get access to error message
    - reference error `except ErrorType as variable_name`
    - then use `variable_name` in block

## 17. Reading and Writing Files
- use `open()`, the returned file `.read()` and `.close()` to read a file
- use the opened files `.write()` method to write to it while it's open
- you can open too many files
- use statement `with open('path/file.txt', 'r') as f` to read and auto-close a file

## 18-19 Quiz & Solution
- `. readline()` to get the next line in an open file, `.seek()` to go to a specific byte
- practice opening and reading names from a text file

## 20-21 Quiz & Solution
- associate descriptions with exception names
- practice understanding how to read and act on error messages

## 22. Importing Local Scripts
- using `import`, importing `as` for custom name and dot notation to access directories
- having a main block for getting code running

## 23. The Standard Library
- default classes, methods, functions available with a base install of the language

## 24-25 Quiz & Solution
- use `math`, `random` and find modules that solve specific problems

## 26. Techniques for Importing Modules
- different variations of imports
- getting objects from modules, naming those objects, naming the module
- getting everything from a module with `*` (avoid!)
- a _package_ is a module containing submodules accessed with dot

## 27. Quiz
- ways to do different method calls based on imports

## 28. Third-Party Libraries
- use `pip` to install other peoples' packages
- `requirements.txt` to install dependencies: `pip install -r requirements.txt`
- examples of third-party packages

## 29. Experimenting with an Interpreter
- using the default IDE vs trying something fancier like IPython

## 30. Online Resources
- suggestions of things to search and ways to search
- what are the most useful search terms to use?
- resources ordered by reliability, with Python Tutorial at the top

## 31-32 Practice & Solution
- save data from file in dictionary, ask for string input to access it

## 33. Conclusion
- you completed the intro to python course!
