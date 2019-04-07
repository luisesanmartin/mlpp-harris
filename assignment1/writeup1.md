# Assignment 1

Author: Luis Eduardo San Martin

```python3 setup
import assignment1
import pandas as pd
```

## Problem 1: Data Acquisition and Analysis

1. Download reported crime data from Chicago open data portal for 2017 and
2018.

**Answer:**

We've set a function in the assignment1.py file to help with this:

```python3 p1a
crimes = assignment1.data_on_crimes()
crimes_df = pd.DataFrame.from_dict(crimes)
```
Now `crimes_df` is the dataframe with the crimes data.

2. Generate summary statistics for the crime reports data including but not limited to number of crimes of each type, how they change over time, and how they are
different by neighborhood. Please use a combination of tables and graphs to
present these summary stats.

**Answer:**