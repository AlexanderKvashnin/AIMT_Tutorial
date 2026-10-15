# Part II — Python

## 0. Why Python?

Python is the standard language for data analysis, machine learning, automation and scripting. Its strengths:

- Extremely readable syntax.
- A huge ecosystem (`numpy`, `pandas`, `scikit-learn`, `matplotlib`).
- Great for prototyping and production.

You typically run it one of three ways:

1. **Interactive shell:** `python` in a terminal.
2. **Scripts:** `python myscript.py`.
3. **Notebooks:** Jupyter (`.ipynb`) — best for exploration and tutorials.

---

## 1. Variables and Data Types

```python
# Numbers
age = 25                 # int
price = 19.99            # float
complex_num = 2 + 3j     # complex

# Strings
name = "Alice"
greeting = 'Hello, world'
multiline = """line1
line2"""

# Booleans
is_student = True
is_admin = False

# None (absence of value)
result = None
```

### Inspecting types

```python
type(age)          # <class 'int'>
isinstance(price, float)   # True
```

### Strings — the essentials

```python
s = "Hello, Python"
print(s.upper())              # HELLO, PYTHON
print(s.lower())              # hello, python
print(s.replace("Python", "World"))
print(len(s))                 # 13
print(s.split(", "))          # ['Hello', 'Python']
print("Py" in s)              # True

# f-strings (the modern way to format)
name = "Alice"
print(f"Hello, {name}! You are {25} years old.")
```

---

## 2. Data Structures

### Lists — ordered, mutable

```python
fruits = ["apple", "banana", "cherry"]
fruits.append("date")            # add to end
fruits.insert(1, "kiwi")         # insert at index
fruits.remove("banana")          # remove by value
last = fruits.pop()              # remove and return last

print(fruits[0])                 # first element
print(fruits[-1])                # last element
print(fruits[1:3])               # slice

fruits.sort()
fruits.sort(reverse=True)
len(fruits)
```

### Tuples — ordered, immutable

```python
point = (3, 4)
x, y = point                     # unpacking
```

### Dictionaries — key-value

```python
person = {"name": "Alice", "age": 30}
person["city"] = "Berlin"        # add a key

print(person["name"])
print(person.get("salary", 0))   # safe get with default
person.keys(), person.values(), person.items()
```

### Sets — unordered, unique

```python
s = {1, 2, 3, 3, 3}
print(s)                         # {1, 2, 3}
s.add(4)
s.discard(2)
a & b, a | b, a - b              # intersection, union, difference
```

### List comprehensions — Python's superpower

```python
squares = [x*x for x in range(10)]
evens   = [x for x in range(20) if x % 2 == 0]
words   = ["hello", "world"]
upper   = [w.upper() for w in words]
```

---

## 3. Control Flow

### Conditions

```python
x = 10
if x > 0:
    print("positive")
elif x == 0:
    print("zero")
else:
    print("negative")

# Ternary
label = "even" if x % 2 == 0 else "odd"
```

### Loops

```python
for i in range(5):           # 0,1,2,3,4
    print(i)

for i in range(1, 6):        # 1..5
    print(i)

for fruit in ["apple", "banana"]:
    print(fruit)

for i, fruit in enumerate(["apple", "banana"]):
    print(i, fruit)

# While
n = 0
while n < 5:
    n += 1

# break, continue
for x in range(10):
    if x == 5: break
    if x % 2: continue
    print(x)
```

---

## 4. Functions

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Alice")
greet("Bob", greeting="Hi")

# *args and **kwargs
def total(*args):
    return sum(args)
total(1, 2, 3)      # 6

def show(**kwargs):
    for k, v in kwargs.items():
        print(k, v)
show(a=1, b=2)

# Lambda (anonymous functions)
square = lambda x: x*x
sorted([3,1,2], key=lambda x: -x)   # [3,2,1]
```

### Docstrings

```python
def add(a, b):
    """Return the sum of a and b."""
    return a + b

help(add)
```

---

## 5. Error Handling

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print("Caught:", e)
except (ValueError, TypeError):
    print("Other problem")
else:
    print("No errors")
finally:
    print("Always runs")
```

Raise your own:

```python
def sqrt(x):
    if x < 0:
        raise ValueError("x must be non-negative")
    return x ** 0.5
```

---

## 6. Files

```python
# Writing
with open("out.txt", "w") as f:
    f.write("hello\n")
    f.writelines(["a\n", "b\n"])

# Reading — the safe, modern way
with open("out.txt", "r") as f:
    text = f.read()

# Line by line
with open("out.txt") as f:
    for line in f:
        print(line.rstrip())
```

`with` automatically closes the file, even if an exception occurs. Always prefer it.

---

## 7. Modules and Imports

```python
import math
math.sqrt(9)

from math import sqrt, pi
sqrt(9)

import numpy as np
np.array([1, 2, 3])

from mymodule import my_function
from mymodule import my_function as mf
```

### Creating your own module

Save `utils.py`:
```python
def double(x):
    return 2 * x
```

Use it:
```python
from utils import double
double(5)
```

---

## 8. Classes (Brief Introduction)

```python
class Material:
    def __init__(self, name, hardness):
        self.name = name
        self.hardness = hardness

    def describe(self):
        return f"{self.name}: H = {self.hardness}"

    def __repr__(self):
        return f"Material({self.name!r}, {self.hardness})"

m = Material("SiC", 34.0)
print(m.describe())
```

Key terms:

- `__init__` — constructor.
- `self` — reference to the instance.
- Methods — functions inside a class.
- `__repr__` — how the object prints itself.

---

## 9. The Big Three: NumPy, Pandas, Matplotlib

### NumPy — fast numerical arrays

```python
import numpy as np

a = np.array([1, 2, 3, 4])
a + 10                 # [11, 12, 13, 14]
a * 2                  # element-wise
a.mean(), a.std(), a.sum()

b = np.array([[1, 2], [3, 4]])
b.shape                # (2, 2)
b.T                    # transpose
b @ b                  # matrix multiplication

np.zeros((3, 3))
np.ones((2, 5))
np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)   # [0, 0.25, 0.5, 0.75, 1]
np.random.seed(42)
np.random.randn(3, 3)
```

### Pandas — tabular data

```python
import pandas as pd

# Load
df = pd.read_csv("data.csv")
df = pd.read_csv("data.txt", sep="\t")

# Inspect
df.head()
df.info()
df.describe()
df.shape
df.columns

# Select
df["col"]
df[["col1", "col2"]]
df.loc[0]              # by label
df.iloc[0]             # by position
df[df["x"] > 5]        # filtering

# Modify
df["new"] = df["a"] + df["b"]
df.rename(columns={"old": "new"}, inplace=True)
df.drop(columns=["unused"], inplace=True)

# Group and aggregate
df.groupby("category")["value"].mean()

# Missing values
df.isna().sum()
df.dropna()
df.fillna(0)

# Save
df.to_csv("out.csv", index=False)
```

### Matplotlib / Seaborn — plotting

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Line plot
plt.plot([1, 2, 3], [4, 5, 6])
plt.xlabel("x"); plt.ylabel("y"); plt.title("Example")
plt.show()

# Scatter
plt.scatter(df["x"], df["y"], alpha=0.6)

# Histogram
df["x"].hist(bins=30)

# Heatmap (with seaborn)
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

# Save figure
plt.savefig("plot.png", dpi=150, bbox_inches="tight")
```

---

## 10. A Tiny End-to-End Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load data
df = pd.read_csv("train_hardness.txt", sep="\t")

# 2. Prepare features and target
X = df.drop(columns=["materials", "Hardness"])
y = df["Hardness"]

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a model
model = LinearRegression().fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² :", r2_score(y_test, y_pred))

# 7. Inspect coefficients
coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print(coefs.head())
```

---

## 11. The Pythonic Habits Cheat Sheet

| Habit | Do | Don't |
|---|---|---|
| Loop over indices | `for i, x in enumerate(xs)` | `for i in range(len(xs))` |
| Build a list | `[f(x) for x in xs]` | `for x in xs: r.append(f(x))` |
| Swap variables | `a, b = b, a` | temp variable |
| String formatting | `f"{x:.2f}"` | `"%0.2f" % x` |
| Open files | `with open(...) as f:` | `f = open(...)` |
| Check `None` | `if x is None:` | `if x == None:` |
| Membership | `if key in dict:` | `if dict.get(key) is not None:` |
| Read CSV | `pd.read_csv(path)` | manual parsing |

---

## 12. Where to Go Next

- **Linux:** `man bash`, "The Linux Command Line" by William Shotts (free PDF).
- **Python:** [docs.python.org](https://docs.python.org/3/tutorial/), "Automate the Boring Stuff with Python" (free online).
- **Data science:** `pandas` user guide, `scikit-learn` tutorials, `matplotlib` gallery.
- **Practice:** start every task by writing the *shell version*, then the *Python version*. Repetition is the fastest teacher.

---

You now have a complete beginner's map of Linux and Python. Read it once straight through, then use it as a lookup — every section is independent enough to answer a specific question when you need it. 🚀
