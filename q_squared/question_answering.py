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
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
qa_tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2").to(device)



def get_subtexts(input,max_tokens,question):
    def get_num_tokens(text):
        return len(qa_tokenizer.encode(text))
    def get_desired_token_windows(num_tokens,max_tokens):
        overlap_size = round(max_tokens / 3)
        left_idxs = [0, num_tokens - max_tokens]
        while left_idxs[-1] - left_idxs[-2] > max_tokens - overlap_size:
            left_idxs.insert(-1, left_idxs[-2] + (max_tokens - overlap_size))
        windows = {i:i+max_tokens for i in left_idxs}
        return windows
    def get_best_word_cuts(desired_token_windows, words_token_count):
        NO_ACTUAL_TOKENS_NUM = 2 #start+end tokens
        def get_string_best_single_cut(token_num):
            if token_num == 0:
                return NO_ACTUAL_TOKENS_NUM
            tokens_num_cut = [i for i in words_token_count if token_num - i >= 0][-1]
            return tokens_num_cut
        word_windows_idxs = {}
        for start_token_num in desired_token_windows:
            start_token_num_cut = get_string_best_single_cut(start_token_num)
            end_token_num_cut = get_string_best_single_cut(desired_token_windows[start_token_num])
            can_shorten_left = start_token_num_cut != NO_ACTUAL_TOKENS_NUM
            can_shorten_right = end_token_num_cut != words_token_count[-1]
            start_word_idx = words_token_count.index(start_token_num_cut)
            end_word_idx = words_token_count.index(end_token_num_cut)
            while end_token_num_cut - start_token_num_cut > max_tokens:
                if can_shorten_left:
                    start_word_idx += 1
                    start_token_num_cut = words_token_count[start_word_idx]
                    if (end_token_num_cut - start_token_num_cut) > max_tokens and can_shorten_right:
                        end_word_idx += -1
                        end_token_num_cut = words_token_count[end_word_idx]
                else:
                    end_word_idx += -1
                    end_token_num_cut = words_token_count[end_word_idx]
            word_windows_idxs[start_word_idx] = end_word_idx
        return word_windows_idxs

    num_question_tokens = get_num_tokens(question)
    max_tokens = max_tokens - num_question_tokens
    num_tokens = get_num_tokens(input)
    if num_tokens <= max_tokens:
        return [input]

    desired_token_windows = get_desired_token_windows(num_tokens,max_tokens)
    words_idxs = [0] + [i for i, j in enumerate(input) if j == ' '] + [len(input)]
    words_token_count = [get_num_tokens(input[:i]) for i in words_idxs]
    word_cut_idxs = get_best_word_cuts(desired_token_windows,words_token_count)
    substrings = [input[words_idxs[i]:words_idxs[word_cut_idxs[i]]] for i in word_cut_idxs]
    return substrings




def get_answer(question, text, max_tokens):  # Code taken from https://huggingface.co/transformers/task_summary.html
    texts = get_subtexts(text, max_tokens, question)
    answers = []
    for sub_text in texts:

        inputs = qa_tokenizer.encode_plus(question, sub_text, add_special_tokens=True, return_tensors="pt").to(device)

        input_ids = inputs["input_ids"].tolist()[0]

        answer_start_scores, answer_end_scores = qa_model(**inputs, return_dict=False)

        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

        ans = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        answers.append(ans)
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


