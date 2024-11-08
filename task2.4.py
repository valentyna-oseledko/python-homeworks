# 4. Зчитування рядка з цифрами
digits = input("Enter 5 digits separated by spaces: ")  # Наприклад: 5 4 3 2 1
digits_list = digits.split()
reversed_digits = list(reversed(digits_list))
result_number = ''.join(reversed_digits)
print("Reversed number:", result_number)
