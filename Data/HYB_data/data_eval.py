import pandas as pd 

df = pd.read_csv("./tickets.csv")


master_dict = dict()
for point in df["category"]:
    if point not in master_dict.keys():
        master_dict[point] = 1
    else: 
        master_dict[point] += 1
print(master_dict)

scaled_dict = dict()
for key, val in master_dict.items():
    scaled_dict[key] = val/500


print(scaled_dict)






