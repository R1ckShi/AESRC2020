# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2019-10-08 15:36:36
# @Function:            
# @Last Modified time: 2020-09-13 19:17:44

import sys
import json

def parse_result(result_label):
    ACCENT_LIST = ["US", "UK", "CHN", "IND", "JPN", "KR", "PT", "RU"]
    ACCENT_NUM = len(ACCENT_LIST)
    utt_nums = [0] * ACCENT_NUM
    correct_nums = [0] * ACCENT_NUM
    
    with open(result_label, 'r') as fd:
        for line in fd.readlines():
            if not line.strip():
                continue
            uttid, hyp = line.split()[:]
            hyp = int(hyp) 
            ref = ACCENT_LIST.index(uttid.split('-')[0])
            utt_nums[ref] += 1
            
            if ref == hyp:
                correct_nums[ref] += 1

    acc_per_accent = [100.0 * correct_nums[i] / utt_nums[i] for i in range(ACCENT_NUM)]
    for i in range(ACCENT_NUM):
        print('{} Accent Accuracy: {:.1f}'.format(ACCENT_LIST[i], acc_per_accent[i]))
    print('Average ACC: {} / {} = {:.1f}'.format(sum(correct_nums), sum(utt_nums), 100.0 * sum(correct_nums) / sum(utt_nums)))

def main():
    json_file = sys.argv[1]
    result_label = sys.argv[2]

    with open(json_file, 'r') as fd, open(result_label, 'w+') as w_fd:
        data = json.load(fd)
        uttid_list = list (data["utts"].keys())
        uttid_list.sort()
        print('There are totally %s utts'%(len(uttid_list)))
        for uttid in uttid_list:
            rec_tokenid_list = data['utts'][uttid]["output"][0]["rec_tokenid"].split()
            rec_tokenid = ' '.join(rec_tokenid_list)
            w_fd.write(uttid + ' ' + rec_tokenid + '\n')

    parse_result(result_label)

if __name__ == '__main__':
    main()
