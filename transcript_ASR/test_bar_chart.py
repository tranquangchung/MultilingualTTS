import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Define Data

data=[["A",40, 36, 38, 35, 40],
      ["B",39, 37, 33, 38, 32],
      ["C",28, 30, 33, 39, 24],
      ["D",40, 40, 35, 29, 35],
      ["E", 28, 25, 16, 27, 30]
     ]
# Plot multiple columns bar chart

df=pd.DataFrame(data,columns=["Name","English","Hindi","Maths", "Science", "Computer"])

df.plot(x="Name", y=["English","Hindi","Maths", "Science", "Computer"], kind="bar",figsize=(9,8))

# Show

plt.show()