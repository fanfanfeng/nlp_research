# create by fanfan on 2019/10/22 0022
lable = ['新闻', '笑话' , '股票', ' 成语', ' 故事', ' 儿歌' , ' 食物营养查询' , ' 电影影讯' , '日历', '菜谱' , '国学'  , '诗词' , ' 汇率' , ' 交通限行'  , '单位换算', '  翻译' ,
         ' 计算器' , ' 亲戚关系计算','有声', '天气', '闹钟提醒','系统设置',"","",""]
name_list = ['余海超','徐丹丹',"余杰","高瑗蔚","裘锋锋"]
import numpy as np
arr = np.arange(45)
checkok = 0
output = []
while checkok == 0:
    np.random.shuffle(arr)
    l = arr.reshape(5,9).tolist()
    for item in l:
        #lab = [ lable[i] for i in item if lable[i] != ""]
        #if len(lab) <4:
        #    checkok = 0
        #    output = []
         #   break
        #checkok = 1
        item = [str(i) for i in item]
        output.append(item)
    checkok = 1
for name,label in zip(name_list,output):

    print(name +":"+ " ".join(label))




