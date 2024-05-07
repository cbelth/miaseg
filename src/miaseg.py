from collections import defaultdict, Counter
import networkx as nx

from paradigm import Paradigm
from model import Model

SUFFIX = 'SUFFIX'
PREFIX = 'PREFIX'

class MIASEG(Model):   
    def train(self, split):
        train = split.train
        self._init_paradigms(train)
        print(f'Num Paradigms: {len(self.paradigms)}')
        self._order_morphemes()

    def segment(self, form, feats, with_analysis=False):
        # if a particular form has never been seen, then segmentation fails
        feats = list(feat for feat in feats if feat in self.meaning_to_form)
        
        root_form = None

        prfxs, sufxs = self._get_affixes(feats)
        if len(prfxs) > 0:
            print(f'--- prfxs: {prfxs}')

        temp_form = f'{form}'

        sufx_ana = list()
        sufx_seg = list()
        for sufx in reversed(sufxs): # iterate over sufxs right-to-left
            # grab possible forms of the sufx, and start with most frequent form
            cand_forms = sorted(self.meaning_to_form[sufx].items(), reverse=True, key=lambda it: (len(it[0]), it[-1]))
            hit = False # track whether we've found a form that matches
            for cand, freq in cand_forms:
                if temp_form.endswith(cand): # check if the candidate form matches
                    sufx_ana.append(sufx)
                    sufx_seg.append(cand)
                    temp_form = temp_form[:-len(cand)]
                    hit = True
                    break
            
            if not hit: # if no match, resort to length
                lens = list(len(cand) for cand, _ in cand_forms)
                most_freq_len = Counter(lens).most_common(1)[0][0]
                sufx_ana.append(sufx)
                sufx_seg.append(temp_form[-most_freq_len:])
                temp_form = temp_form[:-most_freq_len]

        root_form = temp_form

        ana = ['ROOT'] + list(reversed(sufx_ana))
        seg = [root_form] + list(reversed(sufx_seg))

        ana = '-'.join(ana)
        seg = '-'.join(seg)

        if with_analysis:
            return seg, ana
        return seg

    def _get_affixes(self, feats):
        morpheme_order = sorted((self.morpheme_order.index(meaning), meaning) for meaning in feats)

        prfxs = list()
        sufxs = list()
        for idx, meaning in morpheme_order:
            if self.meaning_to_typ[meaning] == PREFIX:
                prfxs.append(meaning)
            else:
                sufxs.append(meaning)
        return prfxs, sufxs
        
    def _init_paradigms(self, train):
        self.paradigms = dict()
        for word in train:
            if word.root not in self.paradigms:
                self.paradigms[word.root] = Paradigm(root=word.root)
            self.paradigms[word.root].add_word(word)

    def _order_morphemes(self) -> None:
        affix_orderings = defaultdict(int) # track the inferred pairwise orderings
        self.meaning_to_form = defaultdict(dict) # track the inferred forms of marked meanings
        self.meaning_to_typ = defaultdict(dict) # track the inferred types of marked meanings (e.g., SUFFIX, PREFIX)

        # look at all words that differ in exactly 1 marked feature
        for par in self.paradigms.values():
            for s1, s2 in par.get_one_diff_pairs():
                # get the morpheme meaning that differs between the two words
                meaning = list(set(s2.feats).difference(s1.feats))[0]
                
                typ, form = self._get_marking_type_from_one_off(s1.form, s2.form), self._get_marking_from_one_off(s1.form, s2.form)
                if typ and form:
                    # tabulate implied affix orderings
                    if typ == SUFFIX: # make a suffix come after any other marked affixes
                        for feat in s1.feats:
                            affix_orderings[(feat, meaning)] += 1 # feat -> morpheme
                    if typ == PREFIX: # make a prefix come before any other marked affixes
                        for feat in s1.feats:
                            affix_orderings[(meaning, feat)] += 1 # morpheme -> feat

                    # tabulate implied form
                    if form not in self.meaning_to_form[meaning]:
                        self.meaning_to_form[meaning][form] = 0
                    self.meaning_to_form[meaning][form] += 1
                    if typ not in self.meaning_to_typ[meaning]:
                        self.meaning_to_typ[meaning][typ] = 0
                    self.meaning_to_typ[meaning][typ] += 1

        for meaning in self.meaning_to_typ.keys():
            self.meaning_to_typ[meaning] = sorted(self.meaning_to_typ[meaning].items(), reverse=True, key=lambda it: it[-1])[0][0]

        morpheme_graph = nx.DiGraph()
        morpheme_graph.add_nodes_from(self.meaning_to_form.keys())
        # add the best infered pairwise orderings
        affix_orderings_list = sorted(affix_orderings.items(), reverse=True, key=lambda it: it[-1])
        for pair, count in affix_orderings_list:
            x, y = pair
            if (x, y) not in affix_orderings: # (y, x) must have been added
                continue
            elif (y, x) not in affix_orderings: # (x, y) can be safely added
                morpheme_graph.add_edge(x, y)
            else: # (x, y) has higher frequency than (y, x)
                morpheme_graph.add_edge(x, y)
                del affix_orderings[(y, x)]
        # enforce the ordering
        try:
            # store global ordering of morphemes
            self.morpheme_graph = morpheme_graph
            self.morpheme_order = list(nx.topological_sort(morpheme_graph))
        except nx.NetworkXUnfeasible as e:
            print(e)
            # check for cycles
            print(f'Cycles in graph: {list(nx.simple_cycles(morpheme_graph))}')

    def _get_marking_type_from_one_off(self, s1: str, s2: str) -> str:
        '''
        :s1: a str that should be mapped from
        :s2: a str that should be mapped to
        '''
        if s2.startswith(s1): # feat marked with suffix
            return SUFFIX
        if s2.endswith(s1): # feat marked with prefix
            return PREFIX
        return None # cannot determine how feat marked
    
    def _get_marking_from_one_off(self, s1: str, s2: str) -> str:
        '''
        :s1: a str that should be mapped from
        :s2: a str that should be mapped to
        '''
        if s2.startswith(s1): # feat marked with suffix
            return f'{s2[len(s1):]}'
        if s2.endswith(s1): # feat marked with prefix
            return f'{s2[:len(s2) - len(s1)]}'
        return None # cannot determine how feat marked