import pickle


def diff(data0, data1):
    data0_0 = 0
    data0_1 = 0
    data1_0 = 0
    data1_1 = 0
    num00 = 0
    num11 = 0
    for i in range(62000):
        data0_0 = data0_0 + int(data0[i] ==0)
        data0_1 = data0_1 + int(data0[i] ==1)
        data1_0 = data1_0 + int(data1[i] ==0)
        data1_1 = data1_1 + int(data1[i] ==1)
        num00 = num00 + int(data0[i] == data1[i] and data0[i] == 0)
        num11 = num11 + int(data0[i] == data1[i] and data0[i] == 1)
    # print("data0_0", data0_0)
    # print("data0_1", data0_1)
    # print("data1_0", data1_0)
    # print("data1_1", data1_1)
    # print("num00", num00)
    # print("num11", num11)
    # print("00 correct percent: ", round(num00/data0_0, 2))
    # print("11 correct percent: ", round(num11/data0_1, 2))
    return round(num00/data0_0, 2), round(num11/data0_1, 2)


for i in range(27):
    f=open('/data-xxx/xxx/IGMS_checkpoints/model_06-08-23:47:46/{}_guide.pickle'.format(i),'rb')
    data0=pickle.load(f)
    f.close()
    f=open('/data-xxx/xxx/IGMS_checkpoints/model_06-08-23:47:46/{}_guide.pickle'.format(i+1),'rb')
    data1=pickle.load(f)
    f.close()
    p1, p2 = diff(data0, data1)
    print(i, "00, 11 correct percent: ", p1, p2)