
import math

x1, y1 = math.radians(39.9075000), math.radians(116.3972300)  # Beijing
x2, y2 = math.radians(50.4546600), math.radians(30.5238000)  # Kyiv

distance = 6371.032 * math.acos(math.sin(x1) * math.sin(x2) + math.cos(x1) * math.cos(x2) * math.cos(y1 - y2))
print(f"{distance: >10.3f} km")
