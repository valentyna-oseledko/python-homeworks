# Словник міст
cities = {
    'Kyiv': {
        'country': 'Ukraine',
        'population': 2800000,
        'fact': 'Capital of Ukraine'
    },
    'Paris': {
        'country': 'France',
        'population': 2140000,
        'fact': 'City of lights'
    },
    'Tokyo': {
        'country': 'Japan',
        'population': 13900000,
        'fact': 'Most populous city in the world'
    }
}

# Виведення інформації
for city, info in cities.items():
    print(f"\nCity: {city}")
    for key, value in info.items():
        print(f"{key.capitalize()}: {value}")
