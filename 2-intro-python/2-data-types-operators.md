# Lesson 2: Data Types and Operators

## 1. Introduction
- these are the building blocks for other programming skills
- two boxes for input (edit code) and output (view results)

## 2. Arithmetic Operators
- typical mathematic symbols for operators
- typical order of operations
- `%` for modulo, `**` for exponentiation, `ˆ` for bitwise XOR
- `//` to round down integer in division (even if answer is negative)

## 3-4 Quiz & Solution
- testing your basic skills with arithmetic operators
- they prefer spaces around plus and minus operators but not multiplication and division
- they prefer no space after function name in function call

## 5. Variables and Assignment Operators
- assignment operator equals sign, variable name on its left, value on its right
- assign multiple variables on one line separated by commas
- give names that are descriptive of their values
- only use letters, numbers, underscores
- do not use reserved words
- snake case your names
- assign with `=`, `+=`, `-=`

## 6-7 Quiz & Solution
- practice assigning and augmenting; I used much `*=`

## 8. Integers and Floats
- two different types based on presence of decimal point
- check types with `type()`
- operation with one or more floats produces a float
- no rounding on type conversion, just add or take off after decimal
- consider when to use each based on whether you need a whole number
- floats are approximations
  - `0.1` is slightly higher than 0.1
  - so `0.1 + 0.1 + 0.1 != 0.3`
- whitespace practices
  - no space between function name and opening parens
  - spaces around operators
  - 79-99 character lines instead of longer lines
  - multiple lines are ok!
- see PEP 8 for more standards

## 9. Quiz
- a few real-life examples of floats vs ints
- try dividing by zero

## 10. Booleans
- values `True` or `False`
- comparison operators `==`, `!=`, `<`, `<=`, `>`, `>=`
- logical operators `and`, `or`, `not`

## 11-12 Quiz & Solution
- do a boolean comparison
- understand the difference between `=` and `==`

## 13. Strings
- `str` is immutable and ordered type
- create between single or double quotes
- `+` to concatenate, `*` to repeat
- slash `\` to escape quote character in string
- access character indexes or get `len()` of a string

## 14-15 Quiz & Solution
- repair a simple invalid strings, answer a question about a return value, format a string and use `len()`

## 16. Type and Conversion
- use `type()` to check an object
- note that, like in `print(type(1))`, the inner function is evaluated then its result is passed to the outer
- special functions are used to work with each type
- change types using conversion functions like `str()`

## 17-18 Quiz & Solution
- guess the type, play with types, sum up total sales from strings of numbers

## 19. String Methods
- operators and functions process data (though functions as operators with names and parens)
- methods as functions associated with specific types of objects (they "belong" to the object)
- string methods include `.title()` for title case, `.lower()` to downcase or `.islower()` to check lowercase
- the `.format()` method allows you to interpolate values into the string: `"1 {} 3 4!".format(2)`

## 20. Quiz: String Methods
- calling string methods on non-strings
- play with string methods from documentation
- format a string

## 21. Split
- break text into an array of strings split by character (default whitespace)
- optionally go up to a limited number of splits (split strings)

## 22-23 Quiz & Solution
- find specific indexes and counts of words in a long poetic string

## 24. Bugs
- understand error messages
- research your error message
- use `print` temporarily to log and debug
- common errors: division by zero, syntax/EOF ("end of file"), type errors
- rely on the web community for help

## 25-26 Conclusion & Summary
- good job, but these are just building blocks so far
- handy charts summarizing types, their constructors, and operators
- consider websites that offer more practice (see links doc)
