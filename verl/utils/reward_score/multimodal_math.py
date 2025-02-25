# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

def parse(answer: str) -> str:
    answer = str(answer)

    res_str = ""
    try:
        float(answer)
        res_str = answer
    except Exception as e:

        answer = answer.replace("<|im_end|>", "").strip()

        # match `A. balabala B. balabala`
        pattern = r'(?<!\w)([A-F])(?=\s|[.)\,]|$)(?:[.)\,]?\s*)(.*?)(?=[\s,]*[A-F](?:[.)\,]?\s*)|$)'
        matches = re.findall(pattern, answer, re.DOTALL)
        if matches:
            options = {key: value.strip() for key, value in matches}
            option_keys = list(sorted(list(options.keys())))
            res_str = ",".join(option_keys)
        else:
            # match `120`, `120.3`, `120e3`, `120F`
            pattern = r"([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?[A-Za-z]*)"
            matches = re.findall(pattern, answer)
            if matches:
                res_str = matches[0]
            else:
                res_str = answer
    return res_str

def verify(answer: str, method = "strict") -> bool:
    if method == "strict":
        pattern = r"^(?:([A-Z](?:,[A-Z])*)|((?:\d+\.\d+|\.\d+|\d+|[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)(?:[A-Za-z]+)?))$"
        match = re.fullmatch(pattern, answer)
        if match:
            return True
        else:
            return False
    elif method == "flexible":
        raise NotImplementedError

def extract_solution(solution_str, method='strict'):
    # this also tests the formatting of the model
    solution = re.search(r"####\s+(.+)", solution_str, re.DOTALL)
    if solution is None:
        final_answer = None
    else:
        final_answer = solution.group(0)
        final_answer = final_answer.replace("####", "").replace("$", "").strip()
        final_answer = parse(final_answer)
    return final_answer


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)

    print("-" * 100)
    print("solution_str:", solution_str)
    print("ground_truth:", ground_truth)
    print("parse answer:", answer)
    print("-" * 100)

    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score