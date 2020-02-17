import gun_prediction
import sys
import importlib

counts = dict()
# print(gun_prediction.gun_name)
for x in range(100):
    if tuple(gun_prediction.gun_name) in counts:
        counts[tuple(gun_prediction.gun_name)] += 1
    else:
        counts[tuple(gun_prediction.gun_name)] = 1
    importlib.reload(gun_prediction)
print(counts)



# for word in words:
#         if word in counts:
#             counts[word] += 1
#         else:
#             counts[word] = 1
