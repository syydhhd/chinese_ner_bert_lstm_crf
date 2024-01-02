import torch
import re
'''
lstm+crf，训练得到的最好macro-f1是0.686。
'''
import random
import json
from model_ref import LstmNerModel
import collections
from typing import List
import cutword

#使用具体的句子验证模型表现，与cutword/cutword/ner.py相同，版本较早，建议使用cutword/cutword/ner.py进行测试
class NER(object):
    def __init__(self, model_path, device, preprocess_data_path):
        self.model_path = model_path
        self.device = device
        
        with open(preprocess_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
        self.id2char = data['id2char']
        self.char2id = data['char2id']
        self.label2id = data['label2id']
        self.id2label = {k: v for v, k in self.label2id.items()}
        self.cutword = cutword.Cutter()
        self.init_model()
    def init_model(self):
        model = LstmNerModel(embedding_size=256, hidden_size=128, vocab_size=len(self.char2id), num_tags=len(self.label2id))
        checkpoint = torch.load(self.model_path)

        # 初始化模型和优化器


        # 处理模型加载时的参数名称匹配问题（如果模型在训练时使用了数据并行）
        if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
            new_state_dict = {}
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:]  # 去除模块前缀
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        model = model.to(self.device)
        self.model = model
        self.model.eval()
        
    def digit_alpha_map(self, text):
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
            if char in digit_and_alpha_map:
                new_words.append(digit_and_alpha_map[char])
            else:
                new_words.append(char)
        return ''.join(new_words)
    def is_digit(self, text):
        return text.isdigit()
    
    def is_hanzi(self, text):
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
        

    def is_english(self, text):
        pattern1 = re.compile(r"^[A-Za-z]+$")
        if re.match(pattern1, text):
            return True
        else:
            return False
    
    def is_special_token(self, text):
        pattern1 = re.compile(r"^[A-Za-z]+$")
        pattern2 = re.compile(r"^[A-Za-z0-9]+$")
        pattern3 = re.compile(r"^[0-9]+$")
        if re.match(pattern1, text):
            return True
        elif not re.match(pattern3, text) and re.match(pattern2, text):
            return True
        else:
            return False
        

    
    def batch_preidct(self, input_list: List[str]):
        input_tensors, seq_lens, input_lists = self.encode(input_list)
        result = self.predict_tags(input_tensors, seq_lens, input_lists)
        return result
    
    
    
    
    def predict_tags(self, input_tensor, seq_lens, input_lists):
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output_fc, mask = self.model(input_tensor, seq_lens)
            predict_tags = self.model.crf.decode(output_fc, mask)
            # print_tag = 0
            # target_tag = random.randint(0, len(true_tags)-1)
            results = []
            for pre, chars in zip(predict_tags, input_lists):
                pre = [self.id2label[t] for t in pre]
                pre_result = self.decode_prediction(chars, pre)
                results.append(pre_result)
            return results
        
    def single_preidct(self, input_str: str):
        input_strs = [input_str]
        return self.batch_preidct(input_strs)[0]
    def encode(self, input_str_list: List[str]):
        input_tensors = []
        seq_lens = []
        input_lists = []
        for input_str in input_str_list:
            
            words = self.cutword.cutword(input_str)
            print(words)
            input_list = []
            for word in words:
                word = word.lower()
                word = self.digit_alpha_map(word)
                if word.strip() == '':
                    for _ in range(len(word.strip())):
                        input_list.append('[SEP]')
                else:

                    for char in word:
                        input_list.append(char)
        
            input_tensor = []
            for char in input_list:
                if char == '[SEP]':
                    continue
                if self.char2id.get(char):
                    input_tensor.append(self.char2id[char])
                else:
                    if self.is_digit(char):
                        input_tensor.append(self.char2id['[NUMBER]'])
                    elif self.is_special_token(char):
                        input_tensor.append(self.char2id['[EWORD]'])
                    elif self.is_hanzi(char):
                        input_tensor.append(self.char2id['[HANZI]'])
                    else:
                        input_tensor.append(self.char2id['[UNK]'])
            seq_len = len(input_tensor)
            input_tensors.append(input_tensor)
            seq_lens.append(seq_len)
            input_lists.append(input_list)
            print()
        return torch.tensor(input_tensors), torch.tensor(seq_lens), input_lists
                
    def decode_prediction(self, chars, tags):
        new_chars = []
        for char in chars:
            if char == '[SEP]':
                new_chars[-1] += ' '
            else:
                new_chars.append(char)
        assert len(new_chars) == len(tags), "{}{}".format(new_chars, tags)
        result = []
        temp = {
            'str':'',
            'begin':-1,
            'end':-1,
            'type':'',            
            }
        
        idx = 0
        for char, tag in zip(new_chars, tags):
            if tag == "O":
                idx += len(char)
                continue
            char_len = len(char)
            head = tag.split('_')[0]
            label = tag.split('_')[-1]
            if "S" in head:
                
                temp['str'] = char
                temp['begin'] = idx
                temp['end'] = idx+1+char_len
                temp['type'] = label
                result.append(temp)
                temp = {
                    'str':'',
                    'begin':-1,
                    'end':-1,
                    'type':''
                    }

            if 'B' in head:
                temp['str'] = char
                temp['begin'] = idx
                temp['type'] = label      
        
                
            elif 'M' in head:
                if not temp['str']:
                    temp = {
                    'str':'',
                    'begin':-1,
                    'end':-1,
                    'type':''
                    }

                    continue
                else:
                    temp['str'] += char
            elif 'E' in head:
                if not temp['str']:
                    temp = {
                    'str':'',
                    'begin':-1,
                    'end':-1,
                    'type':''
                    }
                    continue
                else:
                    temp['str'] += char
                    temp['end'] = idx + char_len
                    result.append(temp)
                    temp = {
                        'str':'',
                        'begin':-1,
                        'end':-1,
                        'type':''
                    }
            else:
                raise Exception("head error")
            idx += char_len
        return result
    
    
    
if __name__ == '__main__':
    
    
    model_path = '/data/chinese_ner_bert_lstm_crf/checkpoint/checkpoint_14_900_0.7950588235294117.pth'
    processed_data = '/data/chinese_ner_bert_lstm_crf/preprocess_data_final.json'
    device = torch.device('cpu')
    ner_model = NER(model_path=model_path, device=device, preprocess_data_path=processed_data)
    sentence_list = []
    sentence = '群众路线活动。'
    for _ in range(10):
        sentence_list.append(sentence)
    result = ner_model.batch_preidct(sentence_list)
    print(result)
    for item in result:
        span_str = item['str']
        print(item['str'])
        begin = item['begin']
        end = item['end']
        print(sentence[begin:end])
        span = sentence[begin:end]
        print(span == span_str)