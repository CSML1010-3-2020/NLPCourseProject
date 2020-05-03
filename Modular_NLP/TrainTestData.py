import pickle

filename = 'X_train_bow'
infile = open(filename,'rb')
X_train_bow = pickle.load(infile)
infile.close()

filename = 'X_test_bow'
infile = open(filename,'rb')
X_test_bow = pickle.load(infile)
infile.close()

filename = 'rm_chi_opt_bow.x_train_sel'
infile = open(filename,'rb')
X_train_bow_chi_opt = pickle.load(infile)
infile.close()

filename = 'rm_chi_opt_bow.x_test_sel'
infile = open(filename,'rb')
X_test_bow_chi_opt = pickle.load(infile)
infile.close()

filename = 'y_train'
infile = open(filename,'rb')
y_train = pickle.load(infile)
infile.close()

filename = 'y_test'
infile = open(filename,'rb')
y_test = pickle.load(infile)
infile.close()