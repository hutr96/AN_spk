d = np.genfromtxt(train_scp, dtype=str)

N = np.shape(d)[0]

files_all = d[:, 1]
labs_noise = d[:, 2]
labs_spk = d[:, 0]