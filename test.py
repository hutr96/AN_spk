# -*- coding: utf-8 -*-
import scipy.io

filename = "/Users/hutr/Documents/Study/毕设/文件/MFCCs/m059_07_004.mat"

data = scipy.io.loadmat(filename)
print 'data:'
print data
print 'whosdata:'
print scipy.io.whosmat(filename)
out = data[scipy.io.whosmat(filename)[0][0]]
print 'out:'
print out
