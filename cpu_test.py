'''
对训练得到的模型进行测试
'''
import random
from data import *
from model_ref import LstmNerModel
import collections

#使用训练时使用的data来获取数据，注意在进行测试时，应将data中的train_path和eval_path修改验证集的路径，并将use_vocab设为True，从而加快数据加载的速度。
train_dev_data = TrainDevData(train_path='/data/chinese_ner_bert_lstm_crf/pos_tag_data_extract/crf_test_total_single_corrected_final_3_new.txt',
                                eval_path='/data/chinese_ner_bert_lstm_crf/pos_tag_data_extract/crf_test_total_single_corrected_final_3_new.txt',
                                use_vocab=True)
id2tag = train_dev_data.id2tag
id2char = train_dev_data.id2char
# train_dataloader = train_dev_data.train_dataloader
eval_dataloader = train_dev_data.eval_dataloader

vocab_size = train_dev_data.vocab_size
num_tags = train_dev_data.num_tags
print('vocab_size:',vocab_size)
print('num_tags:',num_tags)

lr = 0.001
embedding_size = 256
hidden_size = 128
device = torch.device('cuda:1')

model = LstmNerModel(embedding_size=embedding_size, hidden_size=hidden_size, vocab_size=vocab_size, num_tags=num_tags)

checkpoint_path = '/data/chinese_ner_bert_lstm_crf/checkpoint/checkpoint_2_400_0.7854705882352941_new.pth'
checkpoint = torch.load(checkpoint_path)

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
    
model = model.to(device)


# model = model.half()
# model = torch.nn.DataParallel(model)


params = [{"params": [], 'lr': lr}, {'params': [], 'lr': 100 * lr}]
for p in model.named_parameters():
    if "crf" in p[0]:
        params[1]['params'].append(p[1])
    else:
        params[0]['params'].append(p[1])

optimizer = torch.optim.Adam(params)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)



def decode_pre(chars, tags):
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
                pass
            idx += char_len
        return result


def decode_prediction(chars, tags):
    assert len(chars) == len(tags), "{}{}".format(chars, tags)
    result = collections.defaultdict(set)
    entity = ''
    type1 = ''
    for char, tag in zip(chars, tags):
        head = tag.split('_')[0]
        if "S" in head:
            if entity:
                if type1 != '':
                    result[type1].add(entity)
            result[tag.split("_")[1]].add(char)
            type1 = ''
            entity = ''
        elif 'B' in head:
            if entity:
                if type1 != '':
                    result[type1].add(entity)
            entity = char
            type1 = tag.split('_')[1]
        elif 'M' in head:
            type2 = tag.split('_')[1]
            if type1 == type2:
                entity += char
            elif type1 == '':
                entity = ''
        elif 'E' in head:
            type2 = tag.split('_')[1]
            if entity:
                if type1 == type2:
                    entity += char
                else:
                    entity += '[ERROR]'
                if type1 != '':
                    result[type1].add(entity)
                entity = ''
                type1 = ''

        else:
            if entity:
                if type1 != '':
                    result[type1].add(entity)
            entity = ''
    if entity:
        if type1 != '':
            result[type1].add(entity)
    return result
from tqdm import tqdm

def eval(model=model, eval_dataloader=eval_dataloader):
    model.eval()
    result = {}
    with torch.no_grad():
        for index, (input_tensor, true_tags, seq_lens) in tqdm(enumerate(eval_dataloader)):
            input_tensor = input_tensor.to(device)

            true_tags = true_tags.to(device)
            output_fc, mask = model(input_tensor, seq_lens)
            predict_tags = model.crf.decode(output_fc, mask)
            
            true_tags = list(true_tags.cpu().numpy())
            input_tensor = list(input_tensor.cpu().numpy())
            # print_tag = 0
            # target_tag = random.randint(0, len(true_tags)-1)
            for pre, true, input in zip(predict_tags, true_tags, input_tensor):
                pre = [id2tag[t] for t in pre]
                true = [id2tag[t] for t in true]
                chars = [id2char[c] for c in input if c != 0]
                true = true[:len(chars)]
                pre_result = decode_prediction(chars, pre)
                true_result = decode_prediction(chars, true)
                result_pre = decode_pre(chars, pre)
                result_true = decode_pre(chars, true)
                # for item in result_true:
                #     if item['type'] == 'ORG':
                #         if item['str'] == '上海社保':
                #             print("error:",chars)
                #         print("true:",item['str'])
                #         for item_pre in result_pre:
                #             if item_pre['type'] == 'ORG':
                #                 print("pre:",item_pre['str'])
                #             # print(result_pre)
                #         print('***************************************************')                
                for type, cnt in pre_result.items():
                    if type not in result:
                        result[type] = [0, 0, 0]
                    result[type][1] += len(cnt)
                    if type in true_result:
                        result[type][0] += len(pre_result[type] & true_result[type])
                for type, cnt in true_result.items():
                    if type not in result:
                        result[type] = [0, 0, 0]
                    result[type][2] += len(cnt)

             

    for type, (x, y, z) in result.items():
        X, Y, Z = 1e-10, 1e-10, 1e-10
        X += x
        Y += y
        Z += z

        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        result[type].append(round(precision, 3))
        result[type].append(round(recall, 3))
        result[type].append(round(f1, 3))
    result = [(k, v) for k, v in result.items()]
    macro_f1 = sum([v[1][-1] for v in result])/len(result)
    print("macrof1 {}".format(macro_f1))
    result.sort()
    model.train()
    return macro_f1, result

def save_model(model, result, optimizer, num_epoch, num_step):
    # 创建检查点
    checkpoint = {
        
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 可以根据需要添加其他信息，如当前训练轮次等
        'epoch': num_epoch,  # 如果需要保存当前训练轮次
        'step': num_step,  # 如果需要保存当前训练轮次
        'result': result,  # 保存训练结果
        'model_name': 'lstm_crf',
       
    }

    # 指定保存路径和文件名
    checkpoint_path = '/data/chinese_ner_bert_lstm_crf/checkpoint/checkpoint_{}_{}_{}.pth'.format(num_epoch, num_step, result)

    # 保存检查点
    torch.save(checkpoint, checkpoint_path)

import time
# def train(model=model, train_loader=train_dataloader, optimizer=optimizer, scheduler=scheduler, epoch=2):
#     max_result = 0
#     model.train()
#     for i in range(epoch):
#         # start_time = time.time()
#         epoch_loss = 0
#         epoch_count = 0
#         before = -1
#         for index, (input_tensor, tags, seq_lens) in enumerate(train_loader):
#             input_tensor = input_tensor
#             tags = tags
 
#             loss, output_fc, mask = model(input_tensor, tags, seq_lens)
#             loss = torch.mean(loss)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()
#             epoch_count += input_tensor.shape[0]

#             if index % 250 == 0:
#                 print(round(epoch_loss / epoch_count, 3))
#                 cur = epoch_loss / epoch_count
#                 if cur < 0.2 and (before - cur) / before > 0.01:
#                     result = eval(model, eval_dataloader)
#                     print(i, index, 'macrof1 {}'.format(result))
#                 if cur < before:
#                     before = cur
#                 if result > max_result:
#                     max_result = result
#                 save_model(model, result, optimizer, i, index)
                
#         end_time = time.time()
#         # print("epoch {} time {}".format(i, end_time - start_time))

#         scheduler.step()
#     save_model(model, 0, optimizer, epoch, index)

if __name__ == '__main__':

    macro_f1, result = eval()
    print(result)
    print(macro_f1)
