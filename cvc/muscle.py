"""
Wrapper and utility functions for running MUSCLE
"""
import os
import tempfile
import subprocess
import shlex
import logging
from typing import *

import cvc.utils as utils

#返回muscle的比对结果，是有gap(-)对齐后的序列
def run_muscle(sequences: Iterable[str], fast: bool = False) -> List[str]: 
    """
    Run MUSCLE on the given input sequences
    > run_muscle(["DEASV", "KKDEASV", "KKVVVSV"])
    ['--DEASV', 'KKDEASV', 'KKVVVSV']
    """
    with tempfile.TemporaryDirectory() as tempdir:
        #使用 TemporaryDirectory 创建一个临时目录，以便存放输入和输出文件
        # Write the MUSCLE input
        logging.debug(f"Running MUSCLE for MSA in {tempdir}") #输出debug信息到日志
        muscle_input_fname = os.path.join(tempdir, "msa_input.fa")
        with open(muscle_input_fname, "w") as sink:
            sink.write("\n".join([f"> {i}\n{seq}" for i, seq in enumerate(sequences)])) 
            #格式化为fasta并分配索引，返回值为元组
            sink.write("\n")
        # Call MUSCLE
        muscle_output_fname = os.path.join(tempdir, "msa_output.fa")
        muscle_cmd = f"muscle -in {muscle_input_fname} -out {muscle_output_fname}"
            #使用命令行形式调用muscle
        if fast:
            muscle_cmd += " -maxiters 2"
        retval = subprocess.call(
            shlex.split(muscle_cmd),
            stdout=subprocess.DEVNULL,#使用 subprocess.call 执行 MUSCLE 命令，标准输出和错误输出都被重定向到 DEVNULL
            stderr=subprocess.DEVNULL,
        )
        assert retval == 0, f"Exit code {retval} when running muscle"
        msa_seqs = list(utils.read_fasta(muscle_output_fname).values())
    return msa_seqs


def main():
    """On the fly testing"""
    m = run_muscle(["DEASV", "KKDEASV", "KKVVVSV",])
    print(m)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
        #在函数或模块的文档字符串中写下示例，包括输入和预期的输出。然后，使用 doctest.testmod() 来测试这些示例。
    main()
