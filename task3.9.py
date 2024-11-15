# Словник команд NBA
teams = {
    'New York Knicks': [22, 7, 6, 9, 45],
    'Los Angeles Lakers': [20, 10, 5, 5, 50],
    'Chicago Bulls': [18, 8, 7, 3, 40]
}

# Виведення статистики
for team, stats in teams.items():
    print(f"{team.upper()} {stats[0]} {stats[1]} {stats[2]} {stats[3]} {stats[4]}")
