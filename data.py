import torch
from torch.utils.data import DataLoader, Dataset, sampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import random
import json
import os
def digit_alpha_map(text):
    
    digit_and_alpha_map = {
            '１':'1',
            '２':'2',
            '３':'3',
            '４':'4',
            '５':'5',
            '６':'6',
            '７':'7',
            '８':'8',
            '９':'9',
            '０':'0',
            'Ａ':'A',
            'Ｂ':'B',
            'Ｃ':'C',
            'Ｄ':'D',
            'Ｅ':'E',
            'Ｆ':'F',
            'Ｇ':'G',
            'Ｈ':'H',
            'Ｉ':'I',
            'Ｊ':'J',
            'Ｋ':'K',
            'Ｌ':'L',
            'Ｍ':'M',
            'Ｎ':'N',
            'Ｏ':'O',
            'Ｐ':'P',
            'Ｑ':'Q',
            'Ｒ':'R',
            'Ｓ':'S',
            'Ｔ':'T',
            'Ｕ':'U',
            'Ｖ':'V',
            'Ｗ':'W',
            'Ｘ':'X',
            'Ｙ':'Y',
            'Ｚ':'Z',
            'ａ':'a',
            'ｂ':'b',
            'ｃ':'c',
            'ｄ':'d',
            'ｅ':'e',
            'ｆ':'f',
            'ｇ':'g',
            'ｈ':'h',
            'ｉ':'i',
            'ｊ':'j',
            'ｋ':'k',
            'ｌ':'l',
            'ｍ':'m',
            'ｎ':'n',
            'ｏ':'o',
            'ｐ':'p',
            'ｑ':'q',
            'ｒ':'r',
            'ｓ':'s',
            'ｔ':'t',
            'ｕ':'u',
            'ｖ':'v',
            'ｗ':'w',
            'ｘ':'x',
            'ｙ':'y',
            'ｚ':'z',
    }
    new_words = []
    for char in text:
        if digit_and_alpha_map.get(char):
            new_words.append(digit_and_alpha_map[char])
        else:
            new_words.append(char)
    return ''.join(new_words)
def is_digit(text):
    return text.isdigit()

def is_english(text):
    """
    判断序列是否为英文

    :param text: 待判断的序列
    :return: True表示是英文，False表示不是英文
    """
    # 利用正则表达式判断序列是否只包含英文字符
    import re
    pattern1 = re.compile(r"^[A-Za-z]+$")
    pattern2 = re.compile(r"^[A-Za-z0-9]+$")
    pattern3 = re.compile(r"^[0-9]+$")
    if re.match(pattern1, text):
        return True
    elif not re.match(pattern3, text) and re.match(pattern2, text):
        return True
    else:
        return False

def is_hanzi(text):
    """
    判断序列是否为英文

    :param text: 待判断的序列
    :return: True表示是英文，False表示不是英文
    """
    # 利用正则表达式判断序列是否只包含英文字符
    import re
    pattern = re.compile(r"^[\u4e00-\u9fff]*$")
    if re.match(pattern, text):
        return True
    else:
        return False



class CustomNerDataset(Dataset):
    
    r"""
    Args:
        file_path：数据文件路径
        preprocess_data_path: 之前训练后保存的预处理文件
        usual_name_path: 常用姓氏文件路径
        default_vocab_path: 默认词表文件路径：该词表使用频率较高的汉字和英文单词以及字符构成。
        unusual_name_path: 非常用姓氏文件路径
        vocab: 词表
        tags: 标签
        use_vocab: 是否使用词表：通常用于模型评估时加载已有的词表和标签信息
        is_train: 是否为训练集：用于模型训练时是否进行针对‘ORG’标签数据的重采样以及针对不常见姓氏名的构造
    """
    
    def __init__(self, file_path='/data/chinese_ner_bert_lstm_crf/pos_tag_data_extract/crf_train_10000_single.txt', preprocess_data_path=None, usual_name_path=None,default_vocab_path=None, unusual_name_path=None, vocab=None, tags=None, use_vocab=False, is_train=True):
        self.data = []
        self.special_token = {}
        self.num_point = 0
        self.is_train = is_train
        if preprocess_data_path is not None:
            self.preprocess_data_path = preprocess_data_path
        else:
            self.preprocess_data_path = '/data/chinese_ner_bert_lstm_crf/preprocess_data_final_2.json'
        with open(self.preprocess_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.name_list = []
        self.name_list_usual = {}
        if unusual_name_path is not None:
            self.unusual_name_path = unusual_name_path
        else:
            self.unusual_name_path = '/data/chinese_ner_bert_lstm_crf/pos_tag_data_extract/xing.txt'
        with open(self.unusual_name_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                self.name_list.append(word)
        if usual_name_path is not None:
            self.usual_name_path = usual_name_path
        else:
            self.usual_name_path = '/data/chinese_ner_bert_lstm_crf/pos_tag_data_extract/name_usual.txt'
        with open(self.usual_name_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                self.name_list_usual[word] = len(self.name_list_usual)
            
                   
        self.PAD = "[PAD]"
        self.UNK = "[UNK]"
        self.EWORD = "[EWORD]"
        self.NUMBER = "[NUMBER]"
        self.HANZI = "[HANZI]"
        i = 0
        if default_vocab_path is not None:
            self.default_vocab_path = default_vocab_path
        else:
            self.default_vocab_path = '/data/chinese_ner_bert_lstm_crf/pos_tag_data_extract/special_tokens_corrected_2.txt'
        with open(self.default_vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                self.special_token[word] = i
                i += 1
        if use_vocab:
            self.char2id = data['char2id']
        elif vocab:
            self.char2id = vocab
        else:
            self.char2id = {self.PAD: 0, self.UNK: 1, self.EWORD: 2, self.NUMBER: 3, self.HANZI: 4}

        tag2id = data['label2id']
        if use_vocab:
            self.label2id = tag2id
        elif tags:
            self.label2id = tags
        else:
            self.label2id = {'O': 0, 'B_ORG':1, 'M_ORG':2, 'E_ORG':3, 'S_ORG':4}
        self.target_label = {'ORG':0} 
            
        one_example = []
        chars = []
        labels = []
        tags = []
        is_target_label = False
        for line in tqdm(open(file_path)):
            if line == ' \n':
                if len(chars) == 0:
                    print(self.data[-1])
                    print('No chars')
                # 截断
                if len(chars) > 256:
                    chars = chars[:256]
                    labels = labels[:256]

                one_example.append(chars)
                one_example.append(tags)
                one_example.append(labels)
                self.data.append(one_example)
                # 训练集针对self.target_label的重采样
                if is_train:
                    if is_target_label==True:
                        self.data.append(one_example)                        

                is_target_label = False
                chars = []
                labels = []
                tags = []
                one_example = []
            else:
                # 读取每一行数据,并进行划分
                word_tag_label = line.strip().split('\t')
                char = word_tag_label[0]
                label = word_tag_label[-1]
                tag = word_tag_label[1]
                label_type = label.split('_')[-1]
                if self.special_token.get(char):
                    if self.char2id.get(char) != None:
                        pass
                    else:
                        
                        self.char2id[char] = len(self.char2id)
                        
                if self.label2id.get(label):
                    pass
                else:
                    label = 'O'
                if self.target_label.get(label_type) != None:
                    is_target_label = True
                tags.append(tag)
                chars.append(char)
                labels.append(label)
        

        print(len(self.char2id))
        print(len(self.label2id))
        print(len(self.data))
        self.id2char = {v: k for k, v in self.char2id.items()}

                
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chars, tags, labels = self.data[idx]
        assert len(chars) > 0, print('No chars')
        assert len(labels)>0, print('No labels')

        chars_ids = []
        labels_r = []
        is_special = False
        random_float = random.random()
        for char, tag, label in zip(chars, tags, labels):
            #在进行训练的时候，如果数据中包含中文常见姓氏，则以50%的概率随机替换为非常见姓氏
            if label == 'B_PERSON' and tag == 'NR' and self.name_list_usual.get(char) and  random_float > 0.5 and self.is_train:
                        char = random.sample(self.name_list, 1)[0]

                        for i, c in enumerate(char):
                            if self.char2id.get(c) is not None:
                                chars_ids.append(self.char2id[c])
                            else:
                                chars_ids.append(4)
                            if i == 0:
                                labels_r.append(self.label2id['B_PERSON'])
                            else:
                                labels_r.append(self.label2id['M_PERSON'])
                        self.num_point += 1
                        is_special = True
                        
            
            else:
                if self.char2id.get(char):
                    chars_ids.append(self.char2id[char])
                else:
                    if is_hanzi(char):
                        chars_ids.append(4)
                    else:
                        chars_ids.append(1)
                labels_r.append(self.label2id[label])
        
        if is_special and self.num_point < 100:
            print(''.join([self.id2char[i] for i in chars_ids]))
        return {
            'chars': torch.LongTensor(chars_ids),
            'labels': torch.LongTensor(labels_r),
            'len_chars': len(chars_ids)
        }


class TrainDevData:
    def __init__(self, train_path="/data/chinese_ner_bert_lstm_crf/pos_tag_data_extract/crf_train_total_single_corrected_final_3_new.txt",
                 dev_path="/data/chinese_ner_bert_lstm_crf/pos_tag_data_extract/crf_test_total_single_corrected_final_3_new.txt", vocab=None, use_vocab=False, save_preprocess_data=False, preprocess_data_path=None):
        self.train_data = CustomNerDataset(train_path, vocab=vocab, use_vocab=use_vocab, is_train=False)

        self.eval_data = CustomNerDataset(dev_path,
                                          vocab=self.train_data.char2id,
                                          tags=self.train_data.label2id, use_vocab=use_vocab, is_train=False)
        self.id2char = {v: k for k, v in self.train_data.char2id.items()}
        self.id2tag = {v: k for k, v in self.train_data.label2id.items()}

        print(self.train_data.label2id)
        print('*********************************************')
        print(self.id2tag)
        self.vocab_size = len(self.train_data.char2id)
        self.num_tags = len(self.train_data.label2id)

        self.train_dataloader = DataLoader(self.train_data, batch_size=12228,shuffle=True, collate_fn=self.len_collate_fn)
        self.eval_dataloader = DataLoader(self.eval_data, batch_size=12228, collate_fn=self.len_collate_fn)
        #在使用新的数据训练时，使用save_preprocess_data=True，保存预处理的数据
        if save_preprocess_data:
            if preprocess_data_path is None:
                preprocess_data_path = os.path.join(os.path.dirname(train_path), 'preprocess_data.json')
            saved_data = {
                'char2id': self.train_data.char2id,
                'id2char': self.id2char,
                'label2id': self.train_data.label2id,
                'id2label': self.id2tag,
                'vocab_size': self.vocab_size,
                'num_tags': self.num_tags,
            }

            with open(preprocess_data_path, 'w', encoding='utf-8') as f:
                json.dump(saved_data, f, ensure_ascii=False, indent=4)
        


    def len_collate_fn(self, batch_data):
        chars, labels, seq_lens = [], [], []
        for d in batch_data:
            chars.append(d['chars'])
            labels.append(d['labels'])
            seq_lens.append(d['len_chars'])

        chars = pad_sequence(chars, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)
        return chars, labels, torch.LongTensor(seq_lens)


if __name__ == '__main__':
    dataset = CustomNerDataset()
    print(len(dataset.char2id))
    print(len(dataset.label2id))
    print(dataset.data[-1])
