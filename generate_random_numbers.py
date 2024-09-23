import random

count = 1000
print("static const float rands[] = {")
for i in range(count):
    v = random.gauss(0.0, 1.0)
    print(f"    {v},")
print("};")
