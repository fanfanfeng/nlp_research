# create by fanfan on 2019/5/28 0028

def save_xiaohuangji_formatted(data_file):
    print("Processing iaohuangji50w_nofenci.conv...\n")
    conv_path = r'E:\nlp-data\chat\xiaohuangji50w_nofenci.conv'
    pairs = []
    with open(conv_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line != 'E\n']
        # print(content)
    for q, a in zip(lines[::2], lines[1::2]):
        pairs.append([q.strip()[2:].replace('/',' '), a.strip()[2:].replace('/',' ')])

    delimiter = '\t'

    print('Writing to newly formatted file...')
    with open(data_file, 'w', encoding='utf-8') as outputFile:
        for pair in pairs:
            outputFile.write(delimiter.join(pair) + "\n")
    print('Done writing to file, saved as formatted_movie_lines.txt\n')


if __name__ == '__main__':
    save_xiaohuangji_formatted('../data/xiaohuangji.csv')