"""
Featurization code
"""

import os, sys
import logging
import tempfile
from functools import cache, lru_cache
import itertools
import collections
from typing import *
from functools import cached_property
from math import floor

import numpy as np

from transformers import BertTokenizer

import cvc.muscle as muscle
import cvc.utils as utils

#
AA_TRIPLET_TO_SINGLE = {
    "ARG": "R",
    "HIS": "H",
    "LYS": "K",
    "ASP": "D",
    "GLU": "E",
    "SER": "S",
    "THR": "T",
    "ASN": "N",
    "GLN": "Q",
    "CYS": "C",
    "SEC": "U",
    "GLY": "G",
    "PRO": "P",
    "ALA": "A",
    "VAL": "V",
    "ILE": "I",
    "LEU": "L",
    "MET": "M",
    "PHE": "F",
    "TYR": "Y",
    "TRP": "W",
}
AA_SINGLE_TO_TRIPLET = {v: k for k, v in AA_TRIPLET_TO_SINGLE.items()}

# 21 amino acids
AMINO_ACIDS = "RHKDESTNQCUGPAVILMFYW"
assert len(AMINO_ACIDS) == 21
assert all([x == y for x, y in zip(AMINO_ACIDS, AA_TRIPLET_TO_SINGLE.values())])
AMINO_ACIDS_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Pad with $ character
# for single cell model
# PAD = "$"
# MASK = "."
# UNK = "*"
# SEP = "|"
# CLS = "&"
# for bulk trb model
PAD = "$"
MASK = "." #全局变量。在定义后的代码中，函数内还是函数外都可以被访问
UNK = "?"
SEP = "|"
CLS = "*"
AMINO_ACIDS_WITH_ALL_ADDITIONAL = AMINO_ACIDS + PAD + MASK + UNK + SEP + CLS
AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX = {
    aa: i for i, aa in enumerate(AMINO_ACIDS_WITH_ALL_ADDITIONAL)
}


class SequenceMasker: #此函数为对SequenceMasker类的定义
    """Mask one position in each sequence for evaluation (NOT FOR TRAINING)"""

    def __init__(self, seq: Union[str, List[str]], seed: int = 4581):
             # 该init函数为初始化方法，用于创建对象时初始化属性
            #来自于 typing 模块的类型提示表示法。意为是str类型或者由str组成的list都可以。
        self._seed = seed
        self.rng = np.random.default_rng(seed=seed) #np创建随机数
        self.unmasked = [seq] if isinstance(seq, str) else seq
            #isinstance(object, classinfo) 检查对象是否为指定类型/该类型的集合
            #seq是str类型则返回成列表
        self._masked_indices = [] #创建空列表
        self.unmasked_msa = muscle.run_muscle(self.unmasked) #返回多重比对后的结果。包含n(sequence)的相似值
    
    '''
    装饰器，将类的方法转换为一个属性，该属性的值计算一次，然后在实例的生命周期中将其缓存作为普通属性。确保掩码操作只执行一次
    '''
    @cached_property 
    def masked(self) -> List[str]:
        retval = []
        for unmasked in self.unmasked:
            aa = list(unmasked) #aa是列表中的一条序列
            mask_idx = self.rng.integers(0, len(aa)) #在[0,len(aaseq))间，取种子产生的随机一个整数
            assert 0 <= mask_idx < len(aa) #assert条件不满足则报错崩溃。后为条件
            self._masked_indices.append(mask_idx) #mask的位置记录在self对象的_masked_indices属性中
            aa[mask_idx] = MASK #通过下标索引修改列表的值。MASK为变量名，变量值在函数外定义为"."
            retval.append(" ".join(aa))  # Space is necessary for tokenizer
                #将列表中元素用空格连接。 retval存储mask后的序列，
        assert len(self._masked_indices) == len(self)
            #右返回，序列列表的长度，即序列的条数
            #左返回记录序列mask的位置的列表。该语句意即保证for循环中该属性赋值语句执行
        return retval

    @cached_property
    def masked_truth(self) -> List[str]:
        """Return the masked amino acids"""
        _ = self.masked  # Ensure that this has been generated
        return [
            self.unmasked[i][mask_idx] #self.unmasked[i] 为第i条序列的字符串，直接通过双重索引访问被mask的位置
            for i, mask_idx in enumerate(self._masked_indices) #在迭代可迭代对象时，同时获取索引和值
            #这两行为列表推导式：[<expression> for <item> in <iterable>]
            #Python 允许在圆括号、方括号、花括号内直接进行换行，不用使用反斜杠
        ]

    def __len__(self) -> int: #"__xx__"是类的特殊方法。调用len（）后不再使用普通函数，而是这个函数
        return len(self.unmasked)

    def get_naive_predictions(
        self,
        k: int,
        method: Literal[
            "most_common", "random", "most_common_positional"
        ] = "most_common", #Literal[],指定method变量只能包含有限的字面值
    ) -> List[List[str]]:
        """
        Return naive predictions for each of the masked sequences
        Each entry in the list is a list of the top k predictions
        """
        #返回所有未掩码序列加起来最常见的前k个氨基酸。
        if method == "most_common": 
            cnt = collections.Counter() #计数并返回内容和次数
                #cnt形式：({'b': 4, 'a': 3, 'c': 1})
            for seq in self.unmasked:
                cnt.update(seq) #每次循环都将seq的计数信息，更新到cnt对象中。最终是总的计数。
                # 输入字符串，统计单个字符数量
            top_k = [k for k, v in cnt.most_common(k)]  #返回未掩码序列中前k个出现次数最多的氨基酸
            #列表推导式中的变量名k,v只在推导式的上下文中有效。与外部变量无关。与循环式里的变量作用域相同。
            #以列表内嵌套元组的形式：[('b', 4), ('a', 3)]
            return [top_k] * len(self) #创建包含n个top_k列表的大列表，为嵌套列表
            
        #返回每条序列，掩码位置上的pos_most_common（由msa得到）    
        elif method == "most_common_positional":
            # Create a matrix where each row corresponds to a position
            max_len = len(self.unmasked_msa[0]) #多重比对后的序列长度，均相等
            seqs_matrix = np.stack([np.array(list(s)) for s in self.unmasked_msa]).T #所有msa结果储存
            #list()将字符串转换为列表，元素为单个字符
            #生成数组后，np.stack堆叠数组,生成二维数组（[i][j]）
            #转置后，行列互换
            '''
            >>> a
            array([['-', 'C', 'A', 'T', 'T', '-'],
            ['A', 'C', 'G', 'T', '-', '-']], dtype='<U1')
            >>> a.T
            array([['-', 'A'],
            ['C', 'C'],
            ['A', 'G'],
            ['T', 'T'],
            ['T', '-'],
            ['-', '-']], dtype='<U1')
            '''
            
            assert seqs_matrix.shape == (max_len, len(self))
                #确保列数为序列个数，行数为序列长度

            # Per-position predictions
            per_pos_most_common = [] #存储每个位置上最常见的氨基酸
            for i in range(max_len):
                # Excludes padding bases
                cnt = collections.Counter(
                    [aa for aa in seqs_matrix[i] if aa in AMINO_ACIDS]
                ) #对对齐后，每个位置上最常出现的氨基酸进行统计
                per_pos_most_common.append([aa for aa, _n, in cnt.most_common(k)])
                #写入包含i号位前k个aa的列表
            #
            retval = [per_pos_most_common[i] for i in self._masked_indices]
                # 返回序列被掩码位置的pos_most_common氨基酸，按序列顺序
            return retval
            
        elif method == "random":
            baseline_naive_rng = np.random.default_rng(seed=self._seed)
            retval = []
            for _i in range(len(self)):
                idx = [
                    baseline_naive_rng.integers(0, len(AMINO_ACIDS)) for _j in range(k)
                ]
                retval.append([AMINO_ACIDS[i] for i in idx])
            return retval
        else:
            raise ValueError(f"Unrecognized method: {method}")


def adheres_to_vocab(s: str, vocab: str = AMINO_ACIDS) -> bool:
    """
    Returns whether a given string contains only characters from vocab
    >>> adheres_to_vocab("RKDES")
    True
    >>> adheres_to_vocab(AMINO_ACIDS + AMINO_ACIDS)
    True
    """
    return set(s).issubset(set(vocab))


def write_vocab(vocab: Iterable[str], fname: str) -> str:
    """
    Write the vocabulary to the fname, one entry per line
    Mostly for compatibility with transformer BertTokenizer
    """
    with open(fname, "w") as sink:
        for v in vocab:
            sink.write(v + "\n")
    return fname


def get_aa_bert_tokenizer(
    max_len: int = 64, d=AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX
) -> BertTokenizer:
    """
    Tokenizer for amino acid sequences. Not *exactly* the same as BertTokenizer
    but mimics its behavior, encoding start with CLS and ending with SEP

    >>> get_aa_bert_tokenizer(10).encode(insert_whitespace("RKDES"))
    [25, 0, 2, 3, 4, 5, 24]
    """
    with tempfile.TemporaryDirectory() as tempdir:
        vocab_fname = write_vocab(d, os.path.join(tempdir, "vocab.txt"))
        tok = BertTokenizer(
            vocab_fname,
            do_lower_case=False,
            do_basic_tokenize=True,
            tokenize_chinese_chars=False,
            pad_token=PAD,
            mask_token=MASK,
            unk_token=UNK,
            sep_token=SEP,
            cls_token=CLS,
            model_max_len=max_len,
            padding_side="right",
        )
    return tok


def get_pretrained_bert_tokenizer(path: str) -> BertTokenizer:
    """Get the pretrained BERT tokenizer from given path"""
    tok = BertTokenizer.from_pretrained(
        path,
        do_basic_tokenize=False,
        do_lower_case=False,
        tokenize_chinese_chars=False,
        unk_token=UNK,
        sep_token=SEP,
        pad_token=PAD,
        cls_token=CLS,
        mask_token=MASK,
        padding_side="right",
    )

    return tok


def is_whitespaced(seq: str) -> bool:
    """
    Return whether the sequence has whitespace inserted
    >>> is_whitespaced("R K D E S")
    True
    >>> is_whitespaced("RKDES")
    False
    >>> is_whitespaced("R K D ES")
    False
    >>> is_whitespaced("R")
    True
    >>> is_whitespaced("RK")
    False
    >>> is_whitespaced("R K")
    True
    """
    tok = list(seq)
    spaces = [t for t in tok if t.isspace()]
    if len(spaces) == floor(len(seq) / 2):
        return True
    return False


def insert_whitespace(seq: str) -> str:
    """
    Return the sequence of characters with whitespace after each char
    >>> insert_whitespace("RKDES")
    'R K D E S'
    """
    return " ".join(list(seq))


def remove_whitespace(seq: str) -> str:
    """
    Remove whitespace from the given sequence
    >>> remove_whitespace("R K D E S")
    'RKDES'
    >>> remove_whitespace("R K D RR K")
    'RKDRRK'
    >>> remove_whitespace("RKIL")
    'RKIL'
    """
    return "".join(seq.split())


def one_hot(seq: str, alphabet: Optional[str] = AMINO_ACIDS) -> np.ndarray:
    """
    One-hot encode the input string. Since pytorch convolutions expect
    input of (batch, channel, length), we return shape (channel, length)
    When one hot encoding, we ignore the pad characters, encoding them as
    a vector of 0's instead
    """
    if not seq:
        assert alphabet
        return np.zeros((len(alphabet), 1), dtype=np.float32)
    if not alphabet:
        alphabet = utils.dedup(seq)
        logging.info(f"No alphabet given, assuming alphabet of: {alphabet}")
    seq_arr = np.array(list(seq))
    # This implementation naturally ignores the pad character if not provided
    # in the alphabet
    retval = np.stack([seq_arr == char for char in alphabet]).astype(float).T
    assert len(retval) == len(seq), f"Mismatched lengths: {len(seq)} {retval.shape}"
    return retval.astype(np.float32).T


def idx_encode(
    seq: str, alphabet_idx: Dict[str, int] = AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX
) -> np.ndarray:
    """
    Encode the sequence as the indices in the alphabet
    >>> idx_encode("CAFEVVGQLTF")
    array([ 9, 13, 18,  4, 14, 14, 11,  8, 16,  6, 18], dtype=int32)
    """
    retval = np.array([alphabet_idx[aa] for aa in seq], dtype=np.int32)
    return retval


def pad_or_trunc_sequence(seq: str, l: int, right_align: bool = False, pad=PAD) -> str:
    """
    Pad the given sequence to the given length
    >>> pad_or_trunc_sequence("RKDES", 8, right_align=False)
    'RKDES$$$'
    >>> pad_or_trunc_sequence("RKDES", 8, right_align=True)
    '$$$RKDES'
    >>> pad_or_trunc_sequence("RKDESRKRKR", 3, right_align=False)
    'RKD'
    >>> pad_or_trunc_sequence("RKDESRRK", 3, right_align=True)
    'RRK'
    """
    delta = len(seq) - l
    if len(seq) > l:
        if right_align:
            retval = seq[delta:]
        else:
            retval = seq[:-delta]
    elif len(seq) < l:
        insert = pad * np.abs(delta)
        if right_align:
            retval = insert + seq
        else:
            retval = seq + insert
    else:
        retval = seq
    assert len(retval) == l, f"Got mismatched lengths: {len(retval)} {l}"
    return retval


def insert_whitespace(seq: str) -> str:
    """
    Return the sequence of characters with whitespace after each char
    >>> insert_whitespace("RKDES")
    'R K D E S'
    """
    return " ".join(list(seq))


@cache
def all_possible_kmers(alphabet: Iterable[str] = AMINO_ACIDS, k: int = 3) -> List[str]:
    """
    Return all possible kmers
    """
    return ["".join(k) for k in itertools.product(*[alphabet for _ in range(k)])]


@lru_cache(maxsize=128)
def kmer_ft(
    seq: str, k: int = 3, size_norm: bool = False, alphabet: Iterable[str] = AMINO_ACIDS
) -> np.ndarray:
    """
    Kmer featurization to sequence
    """
    kmers = [seq[i : i + k] for i in range(0, len(seq) - k + 1)]
    kmers_to_idx = {
        k: i for i, k in enumerate(all_possible_kmers(alphabet=alphabet, k=k))
    }
    kmers = [k for k in kmers if k in kmers_to_idx]
    idx = np.array([kmers_to_idx[k] for k in kmers])
    retval = np.zeros(len(kmers_to_idx))
    np.add.at(retval, idx, 1)
    assert np.sum(retval) == len(kmers)
    if size_norm:
        retval /= len(kmers)
    return retval
