# Lesson 7: Object-Oriented Programming

## 1. Introduction 
- keep code modular and hide implementation from end user
- example from `scikit-learn`

## 2. Procedural vs. Object-Oriented Programming
- objects have actions and characteristics, like verbs and nouns

## 3. Class, Object, Method and Attribute
- terms for describing object-oriented programming
- class as blueprint, object as instance
- attributes as characteristics, methods as verbs
- encapsulation

## 4. OOP Syntax
- how to write a class in Python
- class name, init method, other methods
- set up attributes in init
- word `self` for referencing the object these methods are living in

## 5. Exercise
- instantiate an object, change attributes, call method

## 6. Notes about OOP
- write getter and setter methods
    - optional since Python doesn't frown on using attributes directly
- underscore before a variable name to indicate it's private
- modularize by having a file with your class in it for importing

## 7. Exercise
- create two classes with attributes and methods, then test them

## 8. Commenting
- reminder to use docstrings explaining class and methods
- explain why and how, args and returns

## 9. Gaussian Class
- formulas for Gaussian and binomial distribution

## 10. How Gaussian Class Works
- explanation of the class they built

## 11. Exercise
- code methods to calculate Gaussian distributions

## 12. Magic Methods
- sum Gaussian distributions together
    - mean: add means together
    - SD: take sqrt of sum of squares of SD
- in code you can't just add two Gaussian objects
    - but Python does have a way to do `guassian_a + gaussian_b`
    - override default behavior using _magic methods_
- magic methods have double underscore before and after
    - they allow custom behavior
    - like the `__add__` magic method overrides the `+` operator
    - or `__repr__` to control what's printed when object alone on a line

## 13. Exercises
- use magic methods in the Gaussian class

## 14. Inheritance
- inherit attributes and methods from parent class
- you could have `Shirt` and `Pants` classes inheriting from `Clothing`
- add attributes and methods
- override any methods

## 15. Exercise
- practice creating child and adding methods to child and parent

## 16. Inheritance: Probability Distribution
- overriding to read data file for distribution

## 17. Demo
- demo of the code with file read

## 18. Advanced Topics
- resources for going beyond classes, objects, attributes, methods, inheritance
- consider class methods, mixins, decorators, multiple inheritance
