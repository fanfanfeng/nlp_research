import os
import re

People2014 = r'E:\BaiduNetdiskDownload\people2014.tar\people2014\2014'

output_file = 'data/sample_text.txt'


def clean_line(text):
    text = text.strip().replace("\n","").replace('\r',"")
    text = re.sub(r'/\S*', "", text)
    text = re.sub(r'\s*', "", text)
    text = text.replace('[',"").replace("]","")
    text = " ".join(list(text)).strip()
    return text

with open(output_file,'w',encoding='utf-8') as fout:
    count = 0
    for root,folders,files in os.walk(People2014):
        for file in files:
            count +=1
            has_data = 0
            real_path = os.path.join(root,file)
            with open(real_path,encoding='utf-8') as fread:
                for line in fread:
                    new_line = clean_line(line)
                    if new_line != "":
                        has_data = 1
                        fout.write(new_line + "\n")
            if has_data:
                fout.write("\n")

        if count > 1000:
            break



