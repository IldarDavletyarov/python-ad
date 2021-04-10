import pandas
import numpy as np
import xlrd, xlwt
import random
import math

RANGE = 600

def get_excel_data():
    rb = xlrd.open_workbook("~/26.xlsx")
    sheet = rb.sheet_by_index(0)
    o = sheet.col_values(4)[1:RANGE+1]
    h = sheet.col_values(5)[1:RANGE+1]
    l = sheet.col_values(6)[1:RANGE+1]
    c = sheet.col_values(7)[1:RANGE+1]
    t = sheet.col_values(3)[1:RANGE+1]
    return t, o, h, l, c

t, o, h, l, c = get_excel_data()

def get_med(x):
    xmed = 0
    for i in range(RANGE):
        xmed = xmed + float(x[i])
    xmed = xmed / RANGE
    return xmed

def get_double_med(x, y):
    xymed = 0
    for i in range(RANGE):
        xymed = xymed + float(y[i]) * float(x[i])
    xymed = xymed / RANGE
    return xymed
    
def get_medsq(x):
    xmedsq = 0
    for i in range(RANGE):
        xmedsq = xmedsq + float(x[i]) * float(x[i])
    xmedsq = xmedsq / RANGE
    return xmedsq

def get_sum_2(x):
    sum = 0
    for i in range(RANGE):
        sum = sum + float(x[i]) * float(x[i])
    return sum    

def get_b(xArray, yArray):
    m = []
    y = []
    for i in range(RANGE):
        newLine = [1]
        for x in xArray:
            newLine.append(float(x[i]))
        m.append(newLine)
        y.append([float(yArray[i])])
    mat = np.matrix(m)
    first = mat.T * mat
    second = first.I
    b = second * mat.T * y
    return b

def get_c_multy_lineal(xArray,e): # function of multi regression C(X1,X2,..., Xn) = B0 +B1*X1 + B2*X2 + BnXn + e
    b = get_b(xArray,c)
    def f(xiArray):
        result = b[0] + e
        for i in range(len(xArray)):
            result += float(b[i+1])*float(xiArray[i])
        return result
    return f

def get_c_multi_polynom(xArray,e):
    result = []
    for j in range(len(xArray)):
        xjArray = xArray[j]
        xjArray = [float(x)**(j+1) for x in xjArray]
        result.append(xjArray)
    b = get_b(result,c)    
    def f(xiArray):
        fResult = b[0] + e # F(X1,X2,Xn) = B0 + B1X + B2X^2 ... + BnX^n + e
        for i in range(len(xiArray)):
            fResult += float(b[i+1])*(float(xiArray[i])**i)
        return fResult    
    return f


def get_r2(x,y):
    xmed = get_med(x)
    ymed = get_med(y)
    xymed = get_double_med(x,y)
    dx = (get_sum_2(x)/len(x)) - xmed*xmed
    dy = (get_sum_2(y)/len(y)) - ymed*ymed
    corr = (xymed - xmed*ymed)/(math.sqrt(dx*dy))
    return corr*corr    

e = 0.01
parameters = [h,o,l]
RANGE_PRINTED = 100

functionP = get_c_multy_lineal(parameters,e)

for i in range(RANGE):
    print("C({},{},{}) = {}".format(h[i],o[i],l[i],functionP([h[i],o[i],l[i]])))


resultP = []
for i in range(RANGE):
    resultP.append(functionP([h[i],o[i],l[i]]))

print(get_r2(o,resultP))

