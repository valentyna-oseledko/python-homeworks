# 6. Візуалізація структури коду
keywords = ('for', 'if', 'else', 'in', ':')

code_structure = """
for each token in the postfix expression :
    if the token is a number :
        print('Convert it to an integer and add it to the end of values')
    else :
        print('Append the result to the end of values')
"""

# Виведення структури коду з правильними відступами
print(code_structure)
