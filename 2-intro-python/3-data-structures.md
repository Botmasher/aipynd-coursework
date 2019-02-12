# Lesson 3: Data Structures

## 1. Introduction
- group and order data types inside data structures

## 2. Lists and Membership Operators
- containers hold other data types or other containers
- `list` as mutable, ordered containers
- lists defined with square brackets
- access list index with square bracketed integer following _zero-based indexing_
- index from the end with negative indices starting with `-1`
- `ListIndexException` received on accessing an index not in the list
- slice with square brackets and colon to get a subsequence from the list
	- lower bound inclusive, upper bound exclusive
	- rest-of notations include omitting starting or ending index
- `in` and `not in` to check membership in list
- mutability and order
	- slicing, indexing and `in` used for both lists and strings
	- but strings cannot be modified unlike lists
	- mutable object: can be modified after creation
	- ordered object: the order of contents matters
	- consider both of these when meeting other containers

## 3-4 Quiz & Solution
- indexing, slicing with negative indexing, determining elements of list after change

## 5. Why Do We Need Lists?
- short example: it can hold a lot of separate strings and you can check if a string is in there

## 6. List Methods
- changing lists after creation and assignment affects all variables pointing to list
- `len()` for counting elements in list
- `max()` and `min()` for greatest/least
	- anything and in the way you can compare with `>`
	- so if you cannot compare values (like integer to string) you can't do it on lists
- `sorted()` for copying the list and returning a sorted copy
- `join()` to concatenate string elements in a list of strings separated by joiner string
- `append()` to push to end of list
- `pop()` to remove last

## 7. Quiz
- practice guessing the output of applying list methods
- playground for testing out list methods

## 8. Check for Understanding: Lists
- multiple-choice quizzes to double check

## 9. Tuples
- immutable and ordered
- parens are optional when making tuples and are often left out when not needed to clarify
	- example that's common: assign multiple values to one variable, then unpack to multiple

## 10. Quiz
- lists vs tuples, play with tuples

## 11. Sets
- container for unique elements that are unordered and mutable
- operators and methods from lists may work, but may have different behaviors
	- `in` checks if the set has a matching element
	- `pop` removes a random element rather than the "last"
- other methods including set-specific and mathematical ones
	- `add` to add to a set
	- `issubset` and others

## 12. Quiz
- if you add an element then pop from a set, the element may or may not be there
- playground test out methods

## 13. Dicts?
- 
