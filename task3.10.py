# Словник речей гравця
things = {
    'key': 3,
    'mace': 1,
    'stone': 24,
    'lantern': 1,
    'gold coin': 10
}

# Виведення інвентарю
print("Equipment:")
total_items = 0
for item, count in things.items():
    print(f"{count} {item}")
    total_items += count

print(f"Total number of things: {total_items}")
