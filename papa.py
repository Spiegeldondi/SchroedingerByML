domain = np.array([0,1,2,3,4,5,6,7,8,9])
func = np.array([3,2,1,5,6,9,3,4,5,6])
steig = []
win_nr = 5



for i in range(len(domain) - 1):
    steig.append(abs(func[i+1] - func[i]))

steig_sum = sum(steig)
goal = steig_sum/win_nr

count = 0
index_list = []

for i in range(len(steig)):
    if count < goal:
        count += steig[i]
    if count >= goal:
        # print("count: ", count)
        print("index: ", i)
        index_list.append(i)
        count = 0
# print("count: ", count)
index_list.append(i)
print("index: ", i)
        
aprx = [0]*len(domain)

aprx[:index_list[0]] = [sum(func[:index_list[0]])/len(func[:index_list[0]])]*len(aprx[:index_list[0]])
for i in range(1, len(index_list)-1):
    aprx[index_list[i]:index_list[i+1]] = [sum(func[index_list[i]:index_list[i+1]])/len(func[index_list[i]:index_list[i+1]])]*len(aprx[index_list[i]:index_list[i+1]])
    
#aprx[:index_list[0]] = [sum(func[:index_list[0]])/len(func[:index_list[0]])]*len(aprx[0:2])
# aprx[index_list[0]:index_list[1]] = [(func[2]+func[3])/2]*len(aprx[2:4])
# aprx[index_list[1]:index_list[2]] = [(func[4])]*len(aprx[4:5])
# aprx[index_list[3]:] = [(func[5]+func[6]+func[7]+func[8]+func[9])/5]*len(aprx[5:]) 

plt.plot(domain,func)
plt.plot(domain,aprx)
