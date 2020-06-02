import csv
import numpy as np
import matplotlib.pyplot as plt

#%%

bins = 128
seedmax = 20 # opens seed files 0 - 19. Lost too much data due to kernel crashes, so these got broken up
trainx = []
trainy = []
validx = []
validy = []

#%%
path = '/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/'
# path = '/home/domi/Dokumente/SchroedingerByML/potentials/D1/'

# %% This is not a ... pythonic [barf]... way of reading data, but python is stupid about pointers, so deal with it
for i in range(seedmax):
    with open(path+'test_pots/test_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainx.append([float(num) for num in row])
    with open(path+'test_out/test_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainy.append([float(num) for num in row])
    with open(path+'valid_pots/valid_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validx.append([float(num) for num in row])
    with open(path+'valid_out/valid_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validy.append([float(num) for num in row])

# %%

# func = np.array([3,2,1,5,6,9,3,4,5,6,1,2,3,4,7,8,9,1,2,5,8,1,13,12,11,10,8,5,4,5]) 
for k in trainx:
    func = k
    slopes = []
    win_nr = 8
    
    
    for i in range(len(func) - 1):
        slopes.append(abs(func[i+1] - func[i]))
    
    steig_sum = sum(slopes)
    goal = steig_sum/win_nr
    
    count = 0
    index_list = []
            
    for i in range(len(slopes)):
        count += slopes[i]
        if (count - goal) >= 0:
            count = count - goal
            index_list.append(i)
        
    aprx = [0]*len(func)
        
    aprx[:index_list[0]+1] = [sum(func[:index_list[0]+1])/len(func[:index_list[0]+1])] * len(func[:index_list[0]+1])
    for i in range(len(index_list)-2):
        aprx[index_list[i]:index_list[i+1]+1] = [sum(func[index_list[i]:index_list[i+1]+1])/len(func[index_list[i]:index_list[i+1]+1])] * len(func[index_list[i]:index_list[i+1]+1])
    
    
    aprx[index_list[len(index_list)-2]:] = [sum(func[index_list[len(index_list)-2]:])/len(func[index_list[len(index_list)-2]:])] * len(func[index_list[len(index_list)-2]:])
    print(len(index_list))
# plt.plot(func)
# plt.plot(aprx)

# for i in range(len(steig)):
#     if count < goal:
#         count += steig[i]
#     if count >= goal:
#         # print("count: ", count)
#         print("index: ", i)
#         index_list.append(i)
#         count = 0
# # print("count: ", count)
# index_list.append(i)
# print("index: ", i)
        
# aprx = [0]*len(domain)

# aprx[:index_list[0]] = [sum(func[:index_list[0]])/len(func[:index_list[0]])]*len(aprx[:index_list[0]])
# for i in range(1, len(index_list)-1):
#     aprx[index_list[i]:index_list[i+1]] = [sum(func[index_list[i]:index_list[i+1]])/len(func[index_list[i]:index_list[i+1]])]*len(aprx[index_list[i]:index_list[i+1]])
    
# #aprx[:index_list[0]] = [sum(func[:index_list[0]])/len(func[:index_list[0]])]*len(aprx[0:2])
# # aprx[index_list[0]:index_list[1]] = [(func[2]+func[3])/2]*len(aprx[2:4])
# # aprx[index_list[1]:index_list[2]] = [(func[4])]*len(aprx[4:5])
# # aprx[index_list[3]:] = [(func[5]+func[6]+func[7]+func[8]+func[9])/5]*len(aprx[5:]) 

# plt.plot(domain,func)
# plt.plot(domain,aprx)
