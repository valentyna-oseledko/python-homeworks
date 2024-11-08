# 2. Введення рядка чисел
numbers = input("Enter a line of integers separated by spaces: ")  # Наприклад: 2 -1 9 6
numbers_list = map(int, numbers.split())
result = sum(numbers_list)
print("The sum of the numbers is:", result)
