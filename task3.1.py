# Глосарій термінів
glossary = {
    "Variable": "A storage location paired with an associated symbolic name.",
    "Function": "A block of code which only runs when it is called.",
    "Loop": "A sequence of instructions that is continually repeated until a condition is met.",
    "Dictionary": "A collection of key-value pairs.",
    "List": "An ordered collection of items which is changeable."
}

# Виведення глосарію
for term, definition in glossary.items():
    print(f"{term}:\n\t{definition}\n")
