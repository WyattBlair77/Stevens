import collections, sys, os
from logic import *

############################################################
# Problem 1: propositional logic
# Convert each of the following natural language sentences into a propositional
# logic formula.  See rainWet() in examples.py for a relevant example.
# sentence: If I have a deadline tomorrow and I'm watching TV, then I'm not being very productive.
def formula1a():
    # Predicates to use:
    tomorrow = Atom('Tomorrow')               # whether it's
    TV = Atom('TV')                 # whether watching TV
    productive = Atom('Productive')               # whether it's productive
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

# sentence: Either I'll go to the gym or go for a run (but not both).
def formula1b():
    # Predicates to use:
    gym = Atom('Gym')     # whether it's gym
    run = Atom('Run') # whether it's night
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

# sentence: The store is open if and only if the sign says "open" and the lights are on.
def formula1c():
    # Predicates to use:
    store = Atom('Store')              # whether it is store
    O = Atom('Open')                # whether it it open
    lights = Atom('Lights')  # whether the lights are on
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 2: first-order logic
# sentence: some people are students, some people are teacher. 
# There exists at least 1 student and 1 teacher, any person must either be a student or a teacher. 
# only student can learn, and only teacher can teach. Every student must have at least one teacher
def formula2a():
    # Predicates to use:
    def Student(x): return Atom('Student', x)
    def Teacher(x): return Atom('Teacher', x)
    def Teaches(x, y): return Atom('Teaches', x, y)
    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")

# sentence: Teacher can also learn from teacher, but student cannot teach
def formula2b():
    # Predicates to use:
    def Student(x): return Atom('Student', x)
    def Teacher(x): return Atom('Teacher', x)
    def Teaches(x, y): return Atom('Teaches', x, y)
    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")


############################################################
# Problem 3: Liar puzzle
# Facts:
# • Adam says: "My shirt is not blue."
# • Levi says: "Adam’s shirt is red."
# • John says: "Levi’s shirt is not blue."
# • Luke says: "John’s shirt is blue.
# • You know that exactly one person is telling the truth 
# • and exactly one person is wearing a red shirt.
# # Query: Who is telling the truth?
# This function returns a list of 6 formulas corresponding to each of the above facts.
# Hint: You might want to use the Equals predicate, defined in logic.py.  
# This predicate is used to assert that two objects are the same.
# In particular, Equals(x,x) = True and Equals(x,y) = False iff x is not equal to y.

def liar():
    def WearsRed(x): return Atom('WearsRed', x)
    def TellTruth(x): return Atom('TellTruth', x)
    luke = Constant('luke')
    john = Constant('john')
    levi = Constant('levi')
    adam = Constant('adam')
    formulas = []
    # We provide the formula for fact 0 here.
    formulas.append(Equiv(TellTruth(adam), WearsRed(adam)))
    # You should add 5 formulas, one for each of facts 1-5.
    # BEGIN_YOUR_CODE 
    
    # END_YOUR_CODE
    query = TellTruth('$x')
    return (formulas, query)

