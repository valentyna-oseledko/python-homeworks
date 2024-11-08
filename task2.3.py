# 3. Список міст
cities = ['Budapest', 'Rome', 'Istanbul', 'Sydney', 'Kyiv', 'Hong Kong']

# Формування повідомлення
message = ', '.join(cities[:-1]) + ' and ' + cities[-1]
print("Formatted message:", message)
