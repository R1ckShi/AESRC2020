#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# Copyright ASLP@NPU. All Rights Reserved
#
# Licensed under the Apache License, Veresion 2.0(the "License");
# You may not use the file except in compliance with the Licese.
# You may obtain a copy of the License at
#
#   http://www.apache.org/license/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author shixian(npu)
# Date 2019/10/09 14:25:50
#
######################################################################
import codecs
import sys
import operator

if __name__ == '__main__':
    filename = sys.argv[1]
    top_nums = int(sys.argv[2])
    prefix = sys.argv[3]
    dict_cn = {}
    dict_en = {}
    f2 = codecs.open("enwords.txt", "w", encoding='utf-8')
    with codecs.open(filename, "r", encoding='utf-8') as f:
        for line in f.readlines():
            if len(line.split('\t')) > 1:
                line = line.split('\t')[1]
                start = 0
            else:
                start = 1
            for char in line.rstrip('\n').split(' ')[start:]:
                if char >= u'\u4e00' and char <= u'\u9fa5':
                    if char not in dict_cn:
                        dict_cn[char] = 1
                    else:
                        dict_cn[char] += 1
                else:
                    f2.write(char + ' ')
                    if char not in dict_en:
                        dict_en[char] = 1
                    else:
                        dict_en[char] += 1
            f2.write('\n')
    dict_cn = sorted(dict_cn.items(),key=operator.itemgetter(1),reverse=True)
    dict_en = sorted(dict_en.items(),key=operator.itemgetter(1),reverse=True)
    fout_cn = codecs.open(prefix + '.cnwf', 'w', encoding='utf-8')
    fout_en = codecs.open(prefix + '.enwf', 'w', encoding='utf-8')
    if len(dict_cn):
        if top_nums == 0:
            for i in range(len(dict_cn)):
                fout_cn.write(dict_cn[i][0] + ' ' + str(dict_cn[i][1]) + '\n')
        else:
            for i in range(top_nums):
                fout_cn.write(dict_cn[i][0] + ' ' + str(dict_cn[i][1]) + '\n')
    if len(dict_en):
        for i in range(len(dict_en)):
            fout_en.write(dict_en[i][0] + ' ' + str(dict_en[i][1]) + '\n')
    fout_cn.close()
    fout_en.close()
    f2.close()
