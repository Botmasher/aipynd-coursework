# Lesson 2: Vectors

## 1. What's a Vector?
- physics: length and direction, on plane or in three dimensions
- cs: ordered list of numbers (and that order matters)
- math: abstract and can contain any numbers
- linear algebra: root vectors at origin
	- numbers then represent the coords

## 2. Vectors part 2
- x and y from origin to tip like [-2 3]
- square brackets for vector, round for points
- z axis then adds third number in triplet like [2 1 3]

## 3. Vectors part 3
- add vector by moving second to end of first then drawing new vector from origin
- it's like taking steps along one vector, then the second, you'll end up there
- calls it "tip-to-tail" method
- this maps really well to adding terms in the numerical representation of coords
- multiply stretches or squishes a vector (negative flips it)
	- "scalars" because they scale
	- "scalar" then is interchangeable with "number" in linear algebra

## 4. Vectors: Mathematical Definition
- vector as ordered list
- each number is a "component" or "coordinate" in field of real numbers
- use steps to calculate movement distance
- use angle to calculate direction (radians or degrees)

## 5. Transpose
- the vector talked about last time is a "column vector"
- tilting this gets you a "row vector"

## 6. Magnitude and Direction
- hypotenuse (use Pythagorean theorem) of vector column is its magnitude
- calculate theta (the `tan^-1` of `y/x`) to get the angle, which is the direction

## 7. Quiz
- magnitude `∥x∥` of a 3D vector `x⃗` getting square root of all three squared

## 8. Operations in the Field
- define operations in field
	- first zero and one element: `[0 0 ... 0 ]` and `[1 1 ... 1]`
- from there addition, multiplication

## 9. Vector Addition
- to add two vectors, add each entry of one to each of other

## 10. Quiz
- practice adding vectors and writing solutions in LaTeX

## 11. Scalar by Vector Multiplication
- multiply every entry in the vector by the scalar

## 12. Quiz
- practice multiplying a vector and writing solution in LaTeX

## 13. Answers
- some logical checks on what scalar variables do to a vector
