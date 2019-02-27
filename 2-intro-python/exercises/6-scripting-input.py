# get and process input for a list of names
names = input("List of names: ").title().split(",")   
# get and process input for a list of the number of assignments
assignments = input("Number of assignments for each name: ").split(",")
# get and process input for a list of grades
grades = input("Grade for each name: ").split(",")

# message string to be used for each student
# HINT: use .format() with this string in your for loop
message = "Hi {},\n\nThis is a reminder that you have {} assignments left to \
submit before you can graduate. Your current grade is {} and can increase \
to {} if you submit all assignments before the due date.\n\n"

# write a for loop that iterates through each set of names, assignments, and grades to print each student's message
# NOTE: consider their potential solution using zip() on the three lists instead
for i, name in enumerate(names):
    print(message.format(names[i], assignments[i], grades[i], int(grades[i]) + (2 * int(assignments[i]))))