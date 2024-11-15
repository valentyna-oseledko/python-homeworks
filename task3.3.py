# Словник мов програмування
languages = {
    'Python': 'Guido van Rossum',
    'Java': 'James Gosling',
    'C++': 'Bjarne Stroustrup',
    'JavaScript': 'Brendan Eich'
}

# Виведення інформації про мови
for lang, developer in languages.items():
    print(f"My favorite programming language is {lang}. It was created by {developer}.")

# Видалення однієї пари та виведення оновленого словника
del languages['C++']
print("\nUpdated dictionary:", languages)
