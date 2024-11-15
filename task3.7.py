# Багаторівневий словник навчальних предметів
subjects = {
    'science': {
        'physics': ['nuclear physics', 'optics', 'thermodynamics'],
        'computer science': {},
        'biology': {}
    },
    'humanities': {},
    'public': {}
}

# Виведення ключів та значень
print("Science subjects:", list(subjects['science'].keys()))
print("Physics topics:", subjects['science']['physics'])
