# Англо-німецький словник
e2g = {
    'stork': 'storch',
    'hawk': 'falke',
    'woodpecker': 'specht',
    'owl': 'eule'
}

# Виведення перекладу слова "owl"
print("German for 'owl' is:", e2g['owl'])

# Додавання нових слів та виведення словника
e2g['eagle'] = 'adler'
e2g['sparrow'] = 'spatz'
print("\nUpdated dictionary:", e2g)
print("Keys:", list(e2g.keys()))
print("Values:", list(e2g.values()))
