# 1. Збереження назв мов
languages = ["Ukrainian", "French", "Bulgarian", "Norwegian", "Latvian"]

# Виведення списку до змін
print("Original list:", languages)

# Використання функції sorted()
sorted_languages = sorted(languages)
print("Sorted list (temporarily):", sorted_languages)
print("Original list after sorted():", languages)

# Використання функції reverse()
languages.reverse()
print("Reversed list:", languages)

# Використання функції sort()
languages.sort()
print("Permanently sorted list:", languages)
