import pandas as pd 
import numpy as np

df = pd.read_csv("./tickets.csv")


aptitude_data = {
    "Network": [
        2,  # Person 1
        3,  # Person 2
        4,  # Person 3
        2,  # Person 4
        3   # Person 5
    ],
    "Software": [
        4,  # Person 1
        5,  # Person 2
        3,  # Person 3
        4,  # Person 4
        2   # Person 5
    ],
    "Account": [
        2,  # Person 1
        3,  # Person 2
        2,  # Person 3
        1,  # Person 4
        2   # Person 5
    ],
    "Training": [
        1,  # Person 1
        2,  # Person 2
        1,  # Person 3
        1,  # Person 4
        2   # Person 5
    ],
    "Security": [
        2,  # Person 1
        3,  # Person 2
        4,  # Person 3
        2,  # Person 4
        3   # Person 5
    ],
    "Licensing": [
        1,  # Person 1
        2,  # Person 2
        1,  # Person 3
        1,  # Person 4
        1   # Person 5
    ],
    "Communication": [
        2,  # Person 1
        3,  # Person 2
        4,  # Person 3
        4,  # Person 4
        3   # Person 5
    ],
    "RemoteWork": [
        2,  # Person 1
        3,  # Person 2
        2,  # Person 3
        2,  # Person 4
        3   # Person 5
    ],
    "Hardware": [
        2,  # Person 1
        2,  # Person 2
        3,  # Person 3
        4,  # Person 4
        2   # Person 5
    ],
    "Infrastructure": [
        1,  # Person 1
        2,  # Person 2
        2,  # Person 3
        1,  # Person 4
        1   # Person 5
    ],
    "Performance": [
        1,  # Person 1
        1,  # Person 2
        2,  # Person 3
        1,  # Person 4
        1   # Person 5
    ]
}
persons = [
    "Alex Johnson",
    "Jessica Smith",
    "Michael Brown",
    "Emily Davis",
    "Daniel Wilson",
    "Sophia Martinez"
]

personal_apt_normed = dict()
for person in range(len(persons)):

    personal_apt_normed[persons[person]] = []
for person in range(len(persons)):
    for apt in aptitude_data.values():
        personal_apt_normed[persons[person]].append(apt[person - 1]/5)

#print(personal_apt_normed)

normalized_data = {}

for name, scores in personal_apt_normed.items():
    min_val = min(scores)
    max_val = max(scores)
    normalized_scores = [(score - min_val) / (max_val - min_val) for score in scores]
    normalized_data[name] = normalized_scores


#print(normalized_data)

final_thing = dict()
counter = 0 
for cat in aptitude_data.keys():
    cur_sub_array = []
    for sub_list in normalized_data.values():
        cur_sub_array.append(sub_list[counter])
    counter += 1
    norm_array = np.array(cur_sub_array) / np.sum(cur_sub_array)

    final_thing[cat] = norm_array
#print(final_thing)

np.random.seed(42)

##for cat in aptitude_data.keys():
    ##sample = np.random.choice(persons, p=final_thing[cat])
    ##print(sample)


df['assigned_worker'] = df.apply(lambda row: np.random.choice(persons, p=final_thing[row["category"]]), axis=1)
#print(df.head())
df.to_csv("tickets_final.csv", index = False)


