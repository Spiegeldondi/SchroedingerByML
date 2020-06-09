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
liste = []
potentials = []
wavefunctions = []
# func = np.array([3,2,1,5,6,9,3,4,5,6,1,2,3,4,7,8,9,1,2,5,8,1,13,12,11,10,8,5,4,5]) 
aprx_list = []
boxInfo = []
index_list_list = []
for k in trainx:
    func = k
    slopes = []
    win_nr = 12
    
    
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
          
    if (len(index_list) == 11):
        liste.append(trainx.index(k))
        potentials.append(k)
        wavefunctions.append(trainy[trainx.index(k)])
        aprx_list.append(aprx)
        index_list_list.append(index_list)
        
        weiten = []
        weiten.append(index_list[0]-1)
        for i in range(1, win_nr-1):
            weiten.append(index_list[i]-index_list[i-1]-1)
        weiten.append(127-index_list[win_nr-2]-1)
        
        positionen = []
        for i in range(win_nr-1):
            positionen.append(index_list[i]-weiten[i]/2)
        positionen.append(127-weiten[win_nr-1]/2)
        
        höhen = []
        höhen.append(aprx[index_list[0]-1])
        for i in index_list:
            höhen.append(aprx[i])
        
        info = []
        
        for i in range(len(höhen)):
            info.append(positionen[i])
            info.append(weiten[i])
            info.append(höhen[i])
            
        boxInfo.append(info)
#%%
X_train = np.vstack(boxInfo[:4000])
y_train = np.vstack(wavefunctions[:4000])
X_valid = np.vstack(boxInfo[4000:])
y_valid = np.vstack(wavefunctions[4000:])
 
#%%
k = trainx[1]
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

# %% elicit descriptors
weiten = []
weiten.append(index_list[0]-1)
for i in range(1, win_nr-1):
    weiten.append(index_list[i]-index_list[i-1]-1)
weiten.append(127-index_list[win_nr-2]-1)

positionen = []
for i in range(win_nr-1):
    positionen.append(index_list[i]-weiten[i]/2)
positionen.append(127-weiten[win_nr-1]/2)

höhen = list(dict.fromkeys(aprx))
#%% choose random sub sample

liste_ind = []
for n in range(100):
    seed = n
    np.random.seed(seed)
    liste_ind.append(np.random.randint(0, len(liste)))

sub_sample_ind = []
for k in liste_ind:
    sub_sample_ind.append(liste[k])

# %%  THIS IS THE SUBSAMPLE
sub_sample = []
for n in sub_sample_ind:
    sub_sample.append(trainx[n])
    
# %% PAPA APRX
pa_aprx = []
for k in sub_sample:
    func = k
    slopes = []
    win_nr = 12
    
    
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
    pa_aprx.append(aprx)

# %% THRESHOLD APRX
    
import copy

tr_aprx = []
for k in sub_sample:
    test = copy.deepcopy(k) 
    
    av = round((max(test)+min(test))/2)
    top = max(test)
    
    for k in test:
        if k <= av:
            test[test.index(k)]=1e-3
        if k > av:
            test[test.index(k)]=top
            
    tr_aprx.append(test)

# %% GD SOLVER
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%%   
tr_aprx_wf = []
pa_aprx_wf = []
bla = 0
for inst in pa_aprx:  # !!! HIER ÄNDERN
    bla += 1
    np.random.seed(seed)
    bins = 128 #dx = 1/bins; actual number of columns saved = bins-1, because 1st and last are 0
    npots = 200 #ends up being 3*this*(validnth-1)/validnth
    validnth = 5 #every nth sample func is saved as validation
    sinval = np.sin([[np.pi*i*j/bins for i in range(1,bins)] for j in range(1,bins//2)])
    cosval = np.cos([[np.pi*i*j/bins for i in range(1,bins)] for j in range(1,bins//2)])
    sqrt2 = np.sqrt(2)
    
    defgrdstate = tf.constant([sqrt2*np.sin(i*np.pi/bins) for i in range(1,bins)])
    psi = tf.Variable(defgrdstate)
    zerotens = tf.zeros([1])
    psil = tf.concat([psi[1:],zerotens],0)
    psir = tf.concat([zerotens,psi[:-1]],0)
    renorm = tf.assign(psi,tf.divide(psi,tf.sqrt(tf.reduce_mean(tf.square(psi)))))
    optimzi = tf.train.GradientDescentOptimizer(0.0625/bins)
    reinit = tf.assign(psi,defgrdstate)
    init = tf.global_variables_initializer()
    
    potentials = []
    validpots = []
    wavefuncs = []
    validfuncs = []
    
    allInfo = []
    boxInfo = []
    k = 0
        
    sess = tf.Session()
    sess.run(init)
    
    i = 1
    j = 1
            
    vofx = copy.deepcopy(inst) 
    vofx = [np.float64(n) for n in vofx]
    
    k += 1
    
    energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx,1.*bins*bins)),
                                        tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))
    training = optimzi.minimize(energy)
    sess.run(reinit)
    
    for t in range(20000):
        sess.run(training)
        sess.run(renorm)

        
    potentials.append(vofx)
    #tr_aprx_wf.append(sess.run(psi).tolist()) # !!! HIER ÄNDERN
    pa_aprx_wf.append(sess.run(psi).tolist())
    print(bla)
# %%
plt.figure(1)
plt.grid(1)
plt.plot(trainx[sub_sample_ind[max_ind]], label="Original Potential")
plt.plot(pa_aprx[max_ind], "--", linewidth=2, label="Processed Potential\n[Threshold 0.5]")
plt.legend() # !!! HIER ÄNDERN
plt.figure(2)
plt.grid(1)
plt.plot(trainy[sub_sample_ind[max_ind]], label="Original Wavefunc")
plt.plot(pa_aprx_wf[max_ind], "--", linewidth=2, label="Processed Wavefunc\n[Threshold 0.5]")
plt.legend() # !!! HIER ÄNDERN

# %%  mse_tr_aprx / mse_pa_aprx
from sklearn.metrics import mean_squared_error as mse

mse_tr_aprx = []
mse_pa_aprx = []

for i in range(len(pa_aprx)): # !!! HIER ÄNDERN
    # mse_tr_aprx.append(mse(trainy[sub_sample_ind[i]], tr_aprx_wf[i]))
    mse_pa_aprx.append(mse(trainy[sub_sample_ind[i]], pa_aprx_wf[i]))
#%%
mse_tr_aprx_sort = copy.deepcopy(mse_tr_aprx)
mse_pa_aprx_sort = copy.deepcopy(mse_pa_aprx)

mse_tr_aprx_sort.sort()
mse_pa_aprx_sort.sort()

# av = mse_tr_aprx_sort[80]   # !!! HIER ÄNDERN
av = mse_pa_aprx_sort[78]

max_ind = mse_pa_aprx.index(max(mse_pa_aprx)) # !!! HIER ÄNDERN
min_ind = mse_pa_aprx.index(min(mse_pa_aprx)) # !!! HIER ÄNDERN
av_ind = mse_pa_aprx.index(av) # !!! HIER ÄNDERN

# %% EXPORT LIST
with open('/home/domi/Schreibtisch/MailMichele/advancedAprx/mse_pa_aprx_sort.csv', 'w') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(mse_tr_aprx_sort)

# %% IMPORT LIST
with open('/home/domi/Schreibtisch/advancedAprx/mse_pa_aprx_sort.csv') as csvfile:
    rd = csv.reader(csvfile)
    train_loss = list(rd)[0]

#%%
plt.figure(1)
plt.grid(1)
plt.title("Error Distribution (100 Instances)")
plt.xlabel("Error [MSE(truth, approx)]")
plt.ylabel("Frequency")
plt.xticks(rotation='vertical')
plt.hist(mse_pa_aprx, bins=50, edgecolor='black')

# %% AVERAGE ERROR
cnt = 0
for k in mse_pa_aprx:   # !!! HIER ÄNDERN
    cnt += k
av_err = cnt/len(mse_pa_aprx)  # !!! HIER ÄNDERN


