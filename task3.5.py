# Словники для домашніх улюбленців
pets = [
    {'name': 'Buddy', 'type': 'dog', 'owner': 'Alex'},
    {'name': 'Whiskers', 'type': 'cat', 'owner': 'Marta'},
    {'name': 'Goldie', 'type': 'fish', 'owner': 'John'}
]

# Виведення інформації
for pet in pets:
    print(f"{pet['owner']} is the owner of a pet - a {pet['type']}.")
