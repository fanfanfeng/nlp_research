# create by fanfan on 2020/3/26 0026
from Competition.datafountain_emotion import settings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set(font_scale=2)

train_df = pd.read_csv(settings.train_origin_data_path,)#engine ='python')
test_df  = pd.read_csv(settings.test_origin_data_path)#,engine ='python')
train_df['情感倾向'].value_counts().plot.bar()
plt.title('sentiment(target)')


train_df = train_df[train_df['情感倾向'].isin(['0','1','-1'])]
print(train_df['微博id'].value_counts().head(10))