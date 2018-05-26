#_*_ coding:utf-8 _*_
import pickle
import os
from functools import reduce

class My_HMM(object):
    """
    Learn model parameters with Maximum Likelihood Estimation, without Baum-Welch algorithm.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.init_prob = {}
        self.trans_prob = {}
        self.emit_prob = {}
        self.char2idx = {'UNK': 0}
        self.states = ['B', 'M', 'E', 'S']

    def get_word_state(self, word):
        if len(word) == 1:
            return ['S']
        word_state = ['B']
        for i in range(1, len(word)):
            if (i+1) < len(word):
                word_state.append('M')
            else:
                word_state.append('E')
        return word_state

    def get_sent_state(self, sent):
        sent_state = []
        seg_line = sent.strip().split()
        for word in seg_line:
            sent_state.extend(self.get_word_state(word))

        return sent_state

    def get_sents_state(self, load=False, path=''):
        if load:
            with open(path, 'rb') as f:
                sents_state = pickle.load(f)
                return sents_state
        else:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                print('size of lines of {} -> {}'.format(self.file_path, len(lines)))
            sents_state = {}
            for i, line in enumerate(lines):
                if len(line) <= 0:
                    continue
                sent_state = []
                seg_line = line.split()
                ori_sent = ''.join(seg_line)
                for word in seg_line:
                    sent_state.extend(self.get_word_state(word))
                sents_state[ori_sent] = sent_state
            print('size of sents_state -> ', len(sents_state))
            with open(path, 'wb') as f:
                pickle.dump(sents_state, f)
            return sents_state

    def get_init_prob(self, load=False, path=''):
        if load:
            with open(path, 'rb') as f:
                self.init_prob = pickle.load(f)
        else:
            sents_state_dict = self.get_sents_state(load=True, path='./Model_MLE/sents_state_dict.pkl')
            init_state_cnt = {'B': 1., 'M': 1., 'E': 1., 'S': 1.}
            total_cnt = 4.0
            for ori_sent, hid_sent in sents_state_dict.items():
                total_cnt += 1
                for i, hid in enumerate(hid_sent):
                    if i == 0:
                        init_state_cnt[hid] += 1
                    else:
                        break
            for key, value in init_state_cnt.items():
                self.init_prob[key] = value / total_cnt
            print(self.init_prob)
            with open(path, 'wb') as f:
                pickle.dump(self.init_prob, f)

    def get_trans_prob(self, load=False, path=''):
        if load:
            with open(path, 'rb') as f:
                self.trans_prob = pickle.load(f)
        else:
            sents_state_dict = self.get_sents_state(load=True, path='./Model_MLE/sents_state_dict.pkl')
            trans_cnt_dict = {}
            count_dict = {}
            for st_1 in self.states:
                count_dict[st_1] = len(self.states)
                cnt_dict = {}
                for st_2 in self.states:
                    cnt_dict[st_2] = 1.0
                trans_cnt_dict[st_1] = cnt_dict

            for ori_sent, hid_sent in sents_state_dict.items():
                for i in range(len(hid_sent)-1):
                    count_dict[hid_sent[i]] += 1.0
                    trans_cnt_dict[hid_sent[i]][hid_sent[i+1]] += 1.0
            for key, cnt_dict in trans_cnt_dict.items():
                prob_dict = {}
                for key_, count in cnt_dict.items():
                    prob_dict[key_] = count / count_dict[key]
                self.trans_prob[key] = prob_dict

            print(self.trans_prob)
            with open(path, 'wb') as f:
                pickle.dump(self.trans_prob, f)

    def get_char2idx(self, load=False, path=''):
        if load:
            with open(path, 'rb') as f:
                self.char2idx = pickle.load(f)
        else:
            sents_state_dict = self.get_sents_state(load=True, path='./Model_MLE/sents_state_dict.pkl')
            for ori_sent, hid_sent in sents_state_dict.items():
                for char in ori_sent:
                    if char not in self.char2idx:
                        self.char2idx[char] = len(self.char2idx)
            print('size of char2idx -> ', len(self.char2idx))
            print('index of 。-> ', self.char2idx["。"])
            print('index of ，-> ', self.char2idx['，'])
            with open(path, 'wb') as f:
                pickle.dump(self.char2idx, f)

    def get_emit_prob(self, load=False, path=''):
        if load:
            with open(path, 'rb') as f:
                self.emit_prob = pickle.load(f)
        else:
            sents_state_dict = self.get_sents_state(load=True, path='./Model_MLE/sents_state_dict.pkl')
            self.get_char2idx(load=True, path='./Model_MLE/char2idx.pkl')
            emit_cnt_dict = {}
            count_dict = {}
            for st in self.states:
                cnt_dict = {}
                count_dict[st] = len(self.char2idx)
                for word,_ in self.char2idx.items():
                    cnt_dict[word] = 1.0
                emit_cnt_dict[st] = cnt_dict

            for ori_sent, hid_sent in sents_state_dict.items():
                for i in range(len(hid_sent)):
                    emit_cnt_dict[hid_sent[i]][ori_sent[i]] += 1
                    count_dict[hid_sent[i]] += 1
            for state, word_cnt in emit_cnt_dict.items():
                prob_dict = {}
                for word, cnt in word_cnt.items():
                    prob_dict[word] = cnt / count_dict[state]
                self.emit_prob[state] = prob_dict
            # print(self.emit_prob)
            with open(path, 'wb') as f:
                pickle.dump(self.emit_prob, f)

    def get_hidden_state(self, sent):
        """
        Viterbi algorithm.
        :param sent:
        :return:
        hidden_state_lst -- the list of hidden states of one sentence.
        """
        sent = sent.strip()
        if len(sent) <= 0:
            return []
        if len(sent) == 1:
            return ['S']

        infer_state_dict = {}
        infer_prob_dict = {}
        for i in range(len(sent)):
            state_dict = {}
            prob_dict = {}
            for state in self.states:
                state_dict[state] = ''
                prob_dict[state] = 0.0
            infer_state_dict[str(i+1)] = state_dict
            infer_prob_dict[str(i+1)] = prob_dict

        for i, char in enumerate(sent):
            if i == 0:
                for st in self.states:
                    try:
                        infer_prob_dict[str(i+1)][st] = self.init_prob[st] * self.emit_prob[st][char]
                    except:
                        infer_prob_dict[str(i+1)][st] = self.init_prob[st] * self.emit_prob[st]['UNK']
            else:
                for st_01 in self.states:
                    max_prob, st_tmp = -1, ''
                    for st_00 in self.states:
                        prob = infer_prob_dict[str(i)][st_00] * self.trans_prob[st_00][st_01]
                        if prob > max_prob:
                            max_prob = prob
                            st_tmp = st_00
                    try:
                        infer_prob_dict[str(i+1)][st_01] = max_prob * self.emit_prob[st_01][char]
                    except:
                        infer_prob_dict[str(i + 1)][st_01] = max_prob * self.emit_prob[st_01]['UNK']
                    infer_state_dict[str(i+1)][st_01] = st_tmp
        hidden_state_lst = []
        last_prob_dict = infer_prob_dict[str(len(sent))]
        last_state = ''
        max_prob = -1
        for state, prob in last_prob_dict.items():
            if prob > max_prob:
                max_prob = prob
                last_state = state
        hidden_state_lst.append(last_state)
        i = len(sent)
        while i > 1:
            stat = infer_state_dict[str(i)][last_state]
            hidden_state_lst.insert(0, stat)
            last_state = stat
            i -= 1
        return hidden_state_lst

    def cut(self, sent):
        sent = sent.strip()
        self.get_init_prob(load=True, path='./Model_MLE/init_prob.pkl')
        self.get_trans_prob(load=True, path='./Model_MLE/trans_prob.pkl')
        self.get_emit_prob(load=True, path='./Model_MLE/emit_prob.pkl')
        hidden_state_lst = self.get_hidden_state(sent)
        print_str = []
        for i in range(len(sent)):
            if hidden_state_lst[i] == 'E' or hidden_state_lst[i] == 'S':
                print_str.append(sent[i])
                print_str.append(' ')
            else:
                print_str.append(sent[i])
        return print_str

    def test(self, seg_lines):
        """
        Test the accuracy of model.
        :param seg_lines:
        :return:
        """
        self.get_init_prob(load=True, path='./Model_MLE/init_prob.pkl')
        self.get_trans_prob(load=True, path='./Model_MLE/trans_prob.pkl')
        self.get_emit_prob(load=True, path='./Model_MLE/emit_prob.pkl')
        if isinstance(seg_lines, str):
            abs_file_path = os.path.abspath(seg_lines)
            if os.path.isfile(abs_file_path):
                with open(abs_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    lines = [line.strip() for line in lines]
            else:
                line = seg_lines.strip()
                lines = [line,]
        elif isinstance(seg_lines, list):
            lines = [line.strip() for line in seg_lines]
        else:
            print('Not valid input.')
            return None
        loss = 0.0
        total = 0.0
        for line in lines:
            sent_state = self.get_sent_state(line)
            total += len(sent_state)
            sent = ''.join(line.split())
            pred_state = self.get_hidden_state(sent)
            if len(sent_state) != len(pred_state):
                loss += 1.0
                print('sent_state -> ', sent_state)
                print('pred_state -> ', pred_state)
            else:
                for i in range(len(sent_state)):
                    if sent_state[i] != pred_state[i]:
                        loss += 1.0
                        # print(line)
                        # print('sent_state -> ', sent_state)
                        # print(' '.join(self.cut(sent)))
                        # print('pred_state -> ', pred_state)
                        # break
        print('average loss of {} lines -> {} %'.format(len(lines), loss / total * 100))

    def score(self, string):
        """
        前向算法, Forward algorithm.
        :param string:
        :return:
        final_prob -- probability of the sentence exists.
        """
        string = string.strip()
        self.get_init_prob(load=True, path='./Model_MLE/init_prob.pkl')
        self.get_trans_prob(load=True, path='./Model_MLE/trans_prob.pkl')
        self.get_emit_prob(load=True, path='./Model_MLE/emit_prob.pkl')
        alpha = []
        for i, char in enumerate(string):
            if i == 0:
                prob = {}
                for state in self.states:
                    try:
                        prob[state] = self.init_prob[state] * self.emit_prob[state][char]
                    except:
                        prob[state] = self.init_prob[state] * self.emit_prob[state]['UNK']
                alpha.append(prob)
            else:
                prob = {}
                for state in self.states:
                    prob[state] = 0.0
                    for st in self.states:
                        prob[state] += alpha[i-1][st] * self.trans_prob[st][state]
                    try:
                        prob[state] *= self.emit_prob[state][char]
                    except:
                        prob[state] *= self.emit_prob[state]['UNK']
                alpha.append(prob)

        final_prob = reduce(lambda x, y: x+y, iter([value for key, value in alpha[-1].items()]))
        # print('probability of {} -> {}'.format(string, final_prob))
        return final_prob



hmm = My_HMM('Corpus/RenMinData.txt_utf8')
# hmm.get_init_prob(load=True, path='Model_MLE/init_prob.pkl')
# hmm.get_trans_prob(load=True, path='./Model_MLE/trans_prob.pkl')
# hmm.get_char2idx(load=False, path='./Model_MLE/char2idx.pkl')
# hmm.get_emit_prob(load=False, path='./Model_MLE/emit_prob.pkl')
sent = '小明的女朋友叫小花。'
print(sent)
print(' '.join(hmm.cut(sent)))
hmm.test('Corpus/test.txt')


