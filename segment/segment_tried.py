# coding=utf-8
class TriedTree:
    trie_tree = {}
    # 初始化: 根据词库建立 trie 树, 需要配合词库文件
    def __init__( self, word_lib_filename = r"E:\git-project\nlp_research/segment/data/crosstalk_storytelling_name.csv" ):
        ifp = open(word_lib_filename,'r',encoding='utf-8')
        word_list = sorted( ifp.readlines() )
        ifp.close()
        for word in word_list:
            cn_chars = list(word.strip())
            if len(cn_chars) <= 1:
                    print("Error for word:%s"%word)
                    continue
            ref = self.trie_tree
            for cn_char in cn_chars:
                    if cn_char not in ref:
                        ref[cn_char] = {}
                    ref = ref[ cn_char ]
            ref[ 'end' ] = True

    # 分词函数, 不去停用词
    def split_to_words(self, content):
        cn_chars = content
        word_list = []
        tmp_search_word = []
        while len( cn_chars ) > 0:
            word_tree = self.trie_tree
            current_word = ""  # 当前词
            search_one_word_success = False
            for (index, cn_char) in enumerate(cn_chars):
                current_word  += cn_char
                if cn_char in word_tree:
                    # 词结束
                    if 'end' in word_tree[cn_char]:
                        if len(word_tree[cn_char]) >1 :
                            tmp_search_word.append((current_word,index))
                            word_tree = word_tree[cn_char]
                        else:
                            word_list.append(current_word)  # 保存当前词
                            search_one_word_success = True
                            break                             # 结束本次搜索
                    # 词未结束
                    else:
                        word_tree = word_tree[cn_char]  # 继续深搜
                # 没有这个字开头的词, 或者这个字与前一个字不能组成词
                else:
                    if len(tmp_search_word)>0:
                        word_list.append(tmp_search_word[-1][0])  # 保存当前词
                        index = tmp_search_word[-1][1]
                        tmp_search_word = []
                        search_one_word_success = True
                    break


            # 第一个字退出, 表示没有以第一个字开头的词
            if search_one_word_success == False:
                cn_chars = cn_chars[1:]
            # 如果不是因为上述原因, 则从下一个字符开始搜索
            elif index+1 < len( cn_chars ):
                cn_chars = cn_chars[index+1:]
            else:
                break
        return word_list

if __name__ == '__main__':
    tied_obj = TriedTree()
    text = input("输入:")
    while text:
        print(tied_obj.split_to_words(text))
        text = input("输入:")



