import itertools

import torch.utils.data


class dataset(torch.utils.data.Dataset):
    def __init__(self, p_src, p_trg, src_max_len=None, trg_max_len=None):
        p_list = [p_src]
        if isinstance(p_trg, str):
            p_list.append(p_trg)
        else:
            p_list.extend(p_trg)
        lines = []
        for p in p_list:
            with open(p) as f:
                lines.append(f.readlines())
        assert len(lines[0]) == len(lines[1])
        self.data = []
        for line in itertools.izip_longest(*lines):
            line = map(lambda v: v.lower().strip(), line)
            if not any(line):
                continue
            line = map(lambda v: v.split(), line)
            if (src_max_len and len(line[0]) > src_max_len) \
                    or (trg_max_len and len(line[1]) > trg_max_len):
                continue
            self.data.append(line)
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index]
