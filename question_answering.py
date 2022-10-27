# Copyright 2020 The Q2 Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, PreTrainedTokenizerFast
from tokenizers import decoders


qa_tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")



def get_subtexts(input,max_tokens,question):
    def get_num_tokens(text):
        return len(qa_tokenizer.encode(text))
    def get_token_idx_windows(num_tokens,max_tokens):
        overlap_size = round(max_tokens / 3)
        left_idxs = [0, num_tokens - max_tokens]
        while left_idxs[-1] - left_idxs[-2] > max_tokens - overlap_size:
            left_idxs.insert(-1, left_idxs[-2] + (max_tokens - overlap_size))
        windows = {i:i+max_tokens for i in left_idxs}
        return windows
    def get_string_best_cut(token_idx):
        if token_idx == 0:
            return 0
        tokens_num_cut = [i for i in words_token_count if token_idx - i >= 0][-1]
        cut_string_idx = words_idxs[words_token_count.index(tokens_num_cut)]
        return cut_string_idx

    if max_tokens == False:
            return [input]
    num_question_tokens = get_num_tokens(question)
    max_tokens = max_tokens - num_question_tokens
    num_tokens = get_num_tokens(input)
    if num_tokens <= max_tokens:
        return [input]

    token_idxs_windows = get_token_idx_windows(num_tokens,max_tokens)
    words_idxs = [i for i, j in enumerate(input) if j == ' '] + [len(input)]
    words_token_count = [get_num_tokens(input[:i]) for i in words_idxs]
    input_cut_idxs = {get_string_best_cut(i):get_string_best_cut(token_idxs_windows[i]) for i in token_idxs_windows}
    substrings = [input[i:input_cut_idxs[i]] for i in input_cut_idxs]
    # print(f'input cut idx: {input_cut_idxs}')
    # print(f'substrings start:end -- {[f"{i[:30]}:{i[-30:]}" for i in substrings]}')
    return substrings


def get_text_windows(text, max_tokens, question):
    def get_num_tokens(text):
        return len(qa_tokenizer.encode(text))

    if max_tokens == False:
        return [text]
    # tokens = qa_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors="pt")
    num_question_tokens = len(qa_tokenizer.encode(question))
    max_tokens = max_tokens - num_question_tokens
    # input_ids = tokens["input_ids"].tolist()[0]
    num_tokens = get_num_tokens(text)
    overlap_size = round(max_tokens / 3)
    if num_tokens <= max_tokens:
        return [text]
    else:
        left_idxs = [0, num_tokens - max_tokens]
        while left_idxs[-1] - left_idxs[-2] > max_tokens - overlap_size:
            left_idxs.insert(-1, left_idxs[-2] + (max_tokens - overlap_size))
        windows = [input_ids[i:i + max_tokens] for i in left_idxs]
    windows_texts = []
    # for window in windows:
    #     string_span = qa_tokenizer.token_to_chars(window[0],window[-1])
    #     string = text[string_span['start']:string_span['end']]
    #     windows_texts.append(string)

    windows_texts = [qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(window)) for window in windows]
    return windows_texts

def get_answer(question, text, max_tokens = False):  # Code taken from https://huggingface.co/transformers/task_summary.html
    texts = get_subtexts(text, max_tokens, question)
    answers = []
    for sub_text in texts:
        inputs = qa_tokenizer.encode_plus(question, sub_text, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        # text_tokens = qa_tokenizer.convert_ids_to_tokens(input_ids)
        # print(f"text: {text}")
        answer_start_scores, answer_end_scores = qa_model(**inputs, return_dict=False)

        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

        ans = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        answers.append(ans)
    # print(answers)
    return answers

# model_name = "ktrapeznikov/albert-xlarge-v2-squad-v2"
#
# nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
#
#
# def get_answer(question, text):
#     QA_input = {
#         'question': question,
#         'context': text
#     }
#     res = nlp(QA_input, handle_impossible_answer=True)
#
#     return res['answer']


