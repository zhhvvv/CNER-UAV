import re
import logging
import zhconv
import pycnnum
import ast
from torch.utils.data import Dataset, DataLoader


class GPTLabelingDataset(Dataset):
    def __init__(self, excel_path: str):
        if excel_path is None:  # 支持测试
            return
        self.data = []
        self.targets = []

        try:
            raw_data = pd.read_csv(excel_path).reset_index(drop=True)
        except:
            raw_data = pd.read_excel(excel_path).reset_index(drop=True)
        raw_data = raw_data.sample(frac=1, random_state=42)  # 乱序
        # raw_data = raw_data.drop_duplicates(subset=['address']).reset_index(drop=True)

        for row in raw_data[['address', 'building', 'unit', 'level', 'room']].itertuples():
            address = self.refine_address(row.address)
            if not address:
                continue

            building = self.refine_label(row.building)
            unit = self.refine_label(row.unit)
            level = self.refine_label(row.level)
            room = self.refine_label(row.room)

            tags = ['O'] * len(address)
            if building: self.tagging_building(address, building, tags)
            if unit: self.tagging_unit(address, unit, tags)
            if level: self.tagging_level(address, level, tags)
            if room: self.tagging_room(address, room, tags)

            self.data.append(list(address))
            self.targets.append(self.iob_tagging(tags))

        logging.info("Load %ds instances from %s", len(self.data), excel_path)

    def iob_tagging(self, tags: list):
        iob_tags = []
        last = 'O'
        for t in tags:
            if t == 'O':
                last = t
                iob_tags.append('O')
                continue

            if t != last:
                iob_tags.append('B-' + t)
            else:
                iob_tags.append('I-' + t)
            last = t

        return iob_tags

    def refine_address(self, address):
        '''
        对用户输入地址预处理，预测时也要执行相同的步骤
        '''
        if pd.isna(address):
            return None

        simple_address = zhconv.convert(address, 'zh-cn')
        return simple_address.lower()

    def refine_label(self, label):
        if label is None or pd.isna(label):
            return None
        return re.sub(r'\s', '', str(label)).lower()

    def tagging_building(self, address: str, building: str, tags: list):
        for b in building.split('|'):
            if self.is_index_span(b):
                self.tag_by_span(b, tags, 'building')
            else:
                self.tag_all_label(address, tags, self.imagine_building(b), 'building')

    def tagging_unit(self, address: str, unit: str, tags: list):
        for u in unit.split('|'):
            if self.is_index_span(u):
                self.tag_by_span(u, tags, 'unit')
            else:
                self.tag_all_label(address, tags, self.imagine_unit(u), 'unit')

    def tagging_level(self, address: str, level: str, tags: list):
        for l in level.split('|'):
            if self.is_index_span(l):
                self.tag_by_span(l, tags, 'level')
            else:
                self.tag_all_label(address, tags, self.imagine_level(l), 'level')

    def tagging_room(self, address: str, room: str, tags: list):
        for r in room.split('|'):
            if self.is_index_span(r):
                self.tag_by_span(r, tags, 'room')
            else:
                for candidate_room in self.imagine_room(r):
                    start = address.find(candidate_room)
                    if start != -1:
                        for i in range(start, start + len(candidate_room)):
                            if tags[i] == 'O':
                                tags[i] = 'room'

    def is_index_span(self, s: str):
        '''
        形如：[n], [n-m]
        '''
        return s.startswith('[') and s.endswith(']')

    def tag_by_span(self, span, tags, tag):
        span = span[1:-1]
        nums = span.split('-')
        start = int(nums[0])
        end = int(nums[1]) if len(nums) > 1 else start
        for i in range(start - 1, end):
            tags[i] = tag

    def tag_all_label(self, address, tags, labels, tag):
        found = False
        for label in labels:
            start = 0
            while start != -1:
                start = address.find(label, start)
                if start != -1:
                    if found and label.isnumeric():
                        # 纯数字只能匹配一次，防止出现类似这样的情况：楼栋是 1，匹配了房间号的 1
                        start += len(label)
                        continue

                    if all(t == 'O' for t in tags[start:start + len(label)]):

                        for i in range(start, start + len(label)):
                            tags[i] = tag

                        found = True

                    start += len(label)

    def imagine_level(self, level: str):
        return self._imagine_suffix(level, ['楼', '层'])

    def imagine_building(self, building: str):
        if building.endswith('号'):
            # xx号 是个门牌号，不认为是楼栋
            return []
        return self._imagine_suffix(building, ['栋', '号楼', '座', '幢'])

    def imagine_unit(self, unit: str):
        return self._imagine_suffix(unit, ['单元'])

    def imagine_room(self, room: str):
        return self._imagine_suffix(room, ['房', '室', '号'])

    def _imagine_suffix(self, item, suffix):
        '''
        对于某个 label，尝试生成不同的后缀
        '''
        alpha = self.find_alpha_prefix(item)
        if alpha is None:
            return [item]

        candidates = [alpha + t for t in suffix]
        if alpha.isdigit() or (alpha[0] == '-' and alpha[1:].isdigit()):
            num = int(alpha)
            if num > -5 and num < 100 and num != 0:  # 其它数字几乎不会出现
                num_cn = self.num2cn(num)
                [candidates.append(num_cn + t) for t in suffix]

        if item not in candidates:
            candidates.append(item)  # 原始的要加载最后，因为优先匹配 xx楼，最后再匹配 xx

        if alpha not in candidates:
            # 地址中写的是 1，标注成 1栋
            candidates.append(alpha)

        candidates.sort(key=len, reverse=True)  # 优先匹配长串
        return candidates

    def num2cn(self, num):
        if num==-1:
            return '负一'
        if num==1:
            return '一'

        if num < 0:
            return '负' + pycnnum.num2cn(-num)

        num_cn = pycnnum.num2cn(num)
        # if 10 <= num and num <= 19:     # 10-19 的数字，库返回的结果会有一个多余的“一”
        #     return num_cn[1:]
        return num_cn

    def find_alpha_prefix(self, s: str) -> str:
        find = re.findall(r'^-?[0-9a-z]+', s)
        if not find:
            return None
        return find[0]

    def isascii(self, s):
        return re.match('^-?[0-9a-zA-Z]+$', s) is not None

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class HumanLabelingDataset(Dataset):
    def __init__(self, excel_path: str):
        if excel_path is None:  # 支持测试
            return
        self.data = []
        self.targets = []

        try:
            raw_data = pd.read_csv(excel_path).reset_index(drop=True)
        except:
            raw_data = pd.read_excel(excel_path).reset_index(drop=True)
        raw_data = raw_data.sample(frac=1, random_state=42)  # 乱序
        # raw_data = raw_data.drop_duplicates(subset=['address']).reset_index(drop=True)

        for row in raw_data[['用户地址', '楼', '单元', '层', '房间']].itertuples():
            address = self.refine_address(row.用户地址)
            if not address:
                continue

            building = self.refine_label(row.楼)
            unit = self.refine_label(row.单元)
            level = self.refine_label(row.层)
            room = self.refine_label(row.房间)

            tags = ['O'] * len(address)
            if building: self.tagging_building(address, building, tags)
            if unit: self.tagging_unit(address, unit, tags)
            if level: self.tagging_level(address, level, tags)
            if room: self.tagging_room(address, room, tags)

            self.data.append(list(address))
            self.targets.append(self.iob_tagging(tags))

        logging.info("Load %ds instances from %s", len(self.data), excel_path)

    def iob_tagging(self, tags: list):
        iob_tags = []
        last = 'O'
        for t in tags:
            if t == 'O':
                last = t
                iob_tags.append('O')
                continue

            if t != last:
                iob_tags.append('B-' + t)
            else:
                iob_tags.append('I-' + t)
            last = t

        return iob_tags

    def refine_address(self, address):
        '''
        对用户输入地址预处理，预测时也要执行相同的步骤
        '''
        if pd.isna(address):
            return None

        simple_address = zhconv.convert(address, 'zh-cn')
        return simple_address.lower()

    def refine_label(self, label):
        if label is None or pd.isna(label):
            return None
        return re.sub(r'\s', '', str(label)).lower()

    def tagging_building(self, address: str, building: str, tags: list):
        for b in building.split('|'):
            if self.is_index_span(b):
                self.tag_by_span(b, tags, 'building')
            else:
                self.tag_all_label(address, tags, self.imagine_building(b), 'building')

    def tagging_unit(self, address: str, unit: str, tags: list):
        for u in unit.split('|'):
            if self.is_index_span(u):
                self.tag_by_span(u, tags, 'unit')
            else:
                self.tag_all_label(address, tags, self.imagine_unit(u), 'unit')

    def tagging_level(self, address: str, level: str, tags: list):
        for l in level.split('|'):
            if self.is_index_span(l):
                self.tag_by_span(l, tags, 'level')
            else:
                self.tag_all_label(address, tags, self.imagine_level(l), 'level')

    def tagging_room(self, address: str, room: str, tags: list):
        for r in room.split('|'):
            if self.is_index_span(r):
                self.tag_by_span(r, tags, 'room')
            else:
                for candidate_room in self.imagine_room(r):
                    start = address.find(candidate_room)
                    if start != -1:
                        for i in range(start, start + len(candidate_room)):
                            if tags[i] == 'O':
                                tags[i] = 'room'

    def is_index_span(self, s: str):
        '''
        形如：[n], [n-m]
        '''
        return s.startswith('[') and s.endswith(']')

    def tag_by_span(self, span, tags, tag):
        span = span[1:-1]
        nums = span.split('-')
        start = int(nums[0])
        end = int(nums[1]) if len(nums) > 1 else start
        for i in range(start - 1, end):
            tags[i] = tag

    def tag_all_label(self, address, tags, labels, tag):
        found = False
        for label in labels:
            start = 0
            while start != -1:
                start = address.find(label, start)
                if start != -1:
                    if found and label.isnumeric():
                        # 纯数字只能匹配一次，防止出现类似这样的情况：楼栋是 1，匹配了房间号的 1
                        start += len(label)
                        continue

                    if all(t == 'O' for t in tags[start:start + len(label)]):
                        for i in range(start, start + len(label)):
                            tags[i] = tag
                        found = True

                    start += len(label)

    def imagine_level(self, level: str):
        return self._imagine_suffix(level, ['楼', '层'])

    def imagine_building(self, building: str):
        if building.endswith('号'):
            # xx号 是个门牌号，不认为是楼栋
            return []
        return self._imagine_suffix(building, ['栋', '号楼', '座', '幢'])

    def imagine_unit(self, unit: str):
        return self._imagine_suffix(unit, ['单元'])

    def imagine_room(self, room: str):
        return self._imagine_suffix(room, ['房', '室', '号'])

    def _imagine_suffix(self, item, suffix):
        '''
        对于某个 label，尝试生成不同的后缀
        '''
        alpha = self.find_alpha_prefix(item)
        if alpha is None:
            return [item]

        candidates = [alpha + t for t in suffix]
        if alpha.isdigit() or (alpha[0] == '-' and alpha[1:].isdigit()):
            num = int(alpha)
            if num > -5 and num < 100 and num != 0:  # 其它数字几乎不会出现
                num_cn = self.num2cn(num)
                [candidates.append(num_cn + t) for t in suffix]

        if item not in candidates:
            candidates.append(item)  # 原始的要加载最后，因为优先匹配 xx楼，最后再匹配 xx

        if alpha not in candidates:
            # 地址中写的是 1，标注成 1栋
            candidates.append(alpha)

        candidates.sort(key=len, reverse=True)  # 优先匹配长串
        return candidates

    def num2cn(self, num):
        if num==-1:
            return '负一'
        if num==1:
            return '一'

        if num < 0:
            return '负' + pycnnum.num2cn(-num)

        num_cn = pycnnum.num2cn(num)
        # if 10 <= num and num <= 19:     # 10-19 的数字，库返回的结果会有一个多余的“一”
        #     return num_cn[1:]
        return num_cn

    def find_alpha_prefix(self, s: str) -> str:
        find = re.findall(r'^-?[0-9a-z]+', s)
        if not find:
            return None
        return find[0]

    def isascii(self, s):
        return re.match('^-?[0-9a-zA-Z]+$', s) is not None

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

