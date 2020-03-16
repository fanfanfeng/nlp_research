# create by fanfan on 2020/3/14 0014
import sys
sys.path.append(r"/data/python_project/nlp_research")
from Competition.datafountain_emotion.model_loader import ModelLoader
from Competition.datafountain_emotion import settings
from Competition.datafountain_emotion.data_process import DataFountainEmotionProcess
import os
def get_best_model_path():
    export_path = os.path.join(settings.Model_save_path,"export/best_exporter")
    max = 0
    for folder in os.listdir(export_path):
        if folder == "." or folder == "..":
            continue
        full_path = os.path.join(export_path,folder)
        if os.path.isdir(full_path):
            if int(folder) > max:
                max = int(folder)

    return os.path.join(export_path,str(max))



def get_submit_file():
    exported_path = get_best_model_path()
    processor = DataFountainEmotionProcess()
    test_examples = processor.get_test_examples(settings.test_data_path)
    model_loader = ModelLoader(exported_path,settings.label_list)
    with open(settings.submit_data_path,'w',encoding='utf-8') as fwrite:
        fwrite.write("id,y\n")
        for index, example in enumerate(test_examples):
            predict_label = model_loader.predict(example.text_a)
            fwrite.write("%s,%s\n" % (example.guid,predict_label) )
            print("预测%s     预测结果为%s" %(example.text_a,predict_label))




