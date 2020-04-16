import gun_prediction

file = open('12GaugePumpActionNoiseModel4.txt','a')
str1 = " "
file.write(str1.join(gun_prediction.gun_name) + "\n")
file.close()

#for i in {1..25}; do python3 Trial2.py; done