# create by fanfan on 2018/8/31 0031
from progress.bar import  Bar
class ProcessBar(Bar):
    message = 'Loading'
    fill = "#"
    suffix = '%(percent).1f%%|ETA:%(eta)ds'
