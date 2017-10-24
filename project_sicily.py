# NLTK imports 
from nltk.corpus import PlaintextCorpusReader 
from nltk import FreqDist, BigramCollocationFinder, Text, ConcordanceIndex
from nltk.collocations import BigramAssocMeasures 

# Ploting grpahs
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import datetime

# Regular expressions
import re

# Various useful libraries
import itertools
from operator import itemgetter
from numpy import log
from collections import defaultdict
from scipy.stats import fisher_exact

########################################
## Utility functions
########################################

def plot_with_dates_no_show(dates, data):

        fig, ax = plt.subplots()
        ax.plot_date(dates, data)

        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.autoscale_view()
        fig.autofmt_xdate()

        ax.fmt_xdata = DateFormatter('%Y-%m-%d')
        ax.grid(True)

def plot_with_dates(dates, data):
        plot_with_dates(dates, data)
        plt.show()

def plot_time_period_no_show(dates, data, D, label):
        return plt.plot(subdivide_freqs_by_date(dates, data, D), label =
                        label)

def plot_time_period(dates, data, D, label = None):
        plot_time_period_no_show(dates, data, D, label)
        plt.show()

def subdivide_freqs_by_date(dates, data, D):
        # Takes a list of dates, a list of frequences, given as a pair
        # (occurrences, total), and number of days. Return a list of
        # frequences for every interval of length <D> starting with the
        # earliest date in <dates>
        dates, data = [list(x) for x in zip(*sorted(zip(dates,
                data)))]
        delta = datetime.timedelta(days = D)
        lower_bound = dates[0] 
        upper_bound = dates[0] + delta
        new_data = []
        i = 0
        while lower_bound <= dates[-1]:
                new_occ = 0
                new_tot = 0
                while i < len(dates) and dates[i] < upper_bound:
                        new_occ += data[i][0]
                        new_tot += data[i][1]
                        i += 1
                if new_tot == 0:
                        new_data.append(0)
                else:
                        new_data.append(new_occ * 1.0 / new_tot)
                lower_bound = upper_bound
                upper_bound = upper_bound + delta
        return new_data

def read_all_corpus(corpus):
        return list(itertools.chain.from_iterable([corpus.words(file) for
                file in corpus.fileids()]))


def get_long_files(corpus, n):
        return [file for file in corpus.fileids() if 
                        len(corpus.words(file)) > n ]

def date_from_file_name(file_name):
        reg = r'([0-9]{4})-([0-9]{2})-([0-9]{2})'
        date = re.search(reg, file_name)
        year, month, day = [int(date.group(i)) for i in range(1, 4)]
        return datetime.date(year, month, day)

########################################
## Token to type ratio
########################################

def type_token_ratio(text, exclude_list):
        # assume that exclude_list is a set
        types = set([token.lower() for token in text]) - exclude_list 
        return float(len(types))/len(text)

def N_most_freq_words(corpus, N):
        tokens = read_all_corpus(corpus)
        fdist = FreqDist([token.lower() for token in tokens])
        return [a for a, b in sorted( fdist.items(), key = itemgetter(1),
                reverse = True)[:N]]


def plot_rare_type_token_ratio(corpus, N, R):
        files = get_long_files(corpus, N) 
        texts = map(lambda x: corpus.words(x), files)
        dates = map(lambda x: date_from_file_name(x), files)

        freq_words = set(N_most_freq_words(corpus, R)) 
        ttratio = map(lambda x: type_token_ratio(x, freq_words), texts )

        plot_with_dates(dates, ttratio) 

########################################
## Frequency of keywords
########################################

# Subdivides the text into chunks of length <N> (ignoring the last chunk if
# it is shorter than <N>) and in each chunk computes the number of
# occurrences of the words in the list <words>. That is the sum of
# occurrences for each of the words.

def freq_evolution(text, words, N):
      
        freqs = []
        for i in range(len(text) / N):
                fq = FreqDist(text[i*N: (i+1)*N])
                num = 0
                for w in words:
                        num += fq[w] 
                freqs.append(num)
        return freqs 

def plot_freq_evolution(text, list_words, N):
        # list_words is a list containing lists of words
        for words in list_words:
                freqs = freq_evolution([token.lower() for token in text],
                                words, N)
                plt.plot(freqs, marker = 'o', linestyle = '-')
        plt.show()

def plot_freq_in_corpus(corpus, list_words):
        # list_words is a list containing lists of words
        freqs = {}
        dates = {}
        
        for i in range(len(list_words)):
                freqs[i] = []
                dates[i] = []
        for f in corpus.fileids():
                text = corpus.words(f) 
                fq = FreqDist(text)
                date = date_from_file_name(f)
                for i in range(len(list_words)):
                        (freqs[i]).append(sum([fq[word] for word in
                                list_words[i]]) * 1.0/len(text))
                        (dates[i]).append(date)

        for i in range(len(list_words)):
                plot_with_dates_no_show(dates[i], freqs[i])
        plt.show()

def plot_freq_evolution_over_time(corpus, list_words, D):
        freqs = {}
        for i in range(len(list_words)):
                freqs[i] = []
        dates = []
        for f in corpus.fileids():
                text = [w.lower() for w in corpus.words(f)]
                fq = FreqDist(text)
                date = date_from_file_name(f)
                dates.append(date)
                for i in range(len(list_words)):
                        freqs[i].append((sum([fq[word] for word in
                                list_words[i]]), len(text)))

        plot_handles = range(len(list_words))
        for i in range(len(list_words)):
                plot_handles[i], = plot_time_period_no_show(dates, freqs[i],
                                D, list_words[i][0])
        plt.legend(plot_handles)
        plt.show()

def N_keyword_evolution_by_date(corpus, D, N, exclude):
        set_exclude = set(exclude)
        files = sorted([ (date_from_file_name(f), f) for f in
                corpus.fileids() ])
        delta = datetime.timedelta(days = D)
        lower_bound = files[0][0] 
        upper_bound = files[0][0] + delta
        keywords = []
        i = 0
        while lower_bound <= files[-1][0]:
                text = []
                while i < len(files) and files[i][0] < upper_bound:
                        new_file = corpus.words(files[i][1])
                        for j in new_file:
                                text.append(j.lower())
                        i += 1
                else:
                        fd = FreqDist(text)
                        new = [] 
                        sort = sorted(fd.items(), key = itemgetter(1),
                                        reverse = True)
                        j = 0
                        while len(new) < N:
                                if not sort[j][0] in set_exclude:
                                        new.append(sort[j][0])
                                j += 1
                keywords.append(new)
                lower_bound = upper_bound
                upper_bound = upper_bound + delta
        return keywords

def N_keyword_evolution_equal_length(corpus, nr_parts, N, exclude):
        set_exclude = set(exclude)
        tokens = [w.lower() for w in read_all_corpus(corpus)]
        L = len(tokens)/nr_parts
        keywords = []
        for i in range(len(tokens)/L): 
                text = tokens[i*L : (i + 1)*L]
                fd = FreqDist(text)
                new = [] 
                sort = sorted(fd.items(), key = itemgetter(1), reverse = True)
                j = 0
                while len(new) < N and j < len(sort):
                        if not sort[j][0] in set_exclude:
                                new.append(sort[j][0])
                        j += 1
                keywords.append(new)
        return keywords

def N_collocations_in_text(text, N, min_freq):
        # finds <N> most significant two word collocations which occur at
        # least <min_freq> times
        text_lower = [w.lower() for w in text]
        finder = BigramCollocationFinder.from_words(text_lower)
        finder.apply_freq_filter(min_freq)
        return finder.nbest(BigramAssocMeasures().pmi, N) 

def plot_freq_evolution_equal_length(corpus, list_words, nr_parts ):
        # divides the corpus into <nr_parts> parts and plots the frequency 
        # of the <lost_words> in each of the segments
       
        tokens = read_all_corpus(corpus)
        N = len(tokens) / nr_parts
        freqs = {}
        for i in range(len(list_words)):
                freqs[i] = []
        for j in range(nr_parts):
                text = [w.lower() for w in tokens[j*N:(j + 1)*N ]]
                fq = FreqDist(text)
                for i in range(len(list_words)):
                        freqs[i].append(sum([fq[word] for word in
                                list_words[i]]))

        plot_handles = range(len(list_words))
        for i in range(len(list_words)):
                plot_handles[i], = plt.plot(freqs[i], label = list_words[i][0])
        plt.legend(plot_handles)
        plt.show()

########################################
## Keyness
########################################

def p_diff(freq, freq_ref):
        # Computes %DIFF for two given frequencies in a form (k, n), where
        # <freq_ref> is the frequency in the reference corpus. For detailed
        # definition of %DIFF see Gabrielatos and Marchi (2012)
        n_freq = float(freq[0]) / freq[1]
        n_freq_ref = float(freq_ref[0]) / freq_ref[1]
        return (n_freq - n_freq_ref) * 100.0 / n_freq_ref 

def dunning_significance(freqs):
        # Takes as input a list containing two frequencies of the form (k , n),
        # where k is the number of occurencies of a given word and n is the
        # length of the text in consideration. The function the computes the
        # significance ofthe difference in the frequency, using log-likelihood
        # measure by Dunning (1993)

        freq_1 = freqs[0]
        freq_2 = freqs[1]
        def log_z(x):
                if x == 0:
                        return 0
                else:
                        return log(x)
        k_1, n_1 = freq_1
        k_2, n_2 = freq_2
        K = k_1 + k_2
        N = n_1 + n_2
        return 2.0 * ( k_1 * log_z(k_1) + k_2 * log_z(k_2) - K * log_z(K) 
                     - n_1 * log_z(n_1) - n_2 * log_z(n_2) + N * log_z(N) 
                     + (n_1 - k_1) * log_z(n_1 - k_1) + (n_2 - k_2) *
                     log_z(n_2 - k_2) - (N - K) * log_z(N - K) )

def N_keywords(text, corpus_ref, N, cut_off, sig_token='LL'):
        # The function finds <N> significant keywords in the text <text> with
        # respect to the reference corpus. The variable <cut_off> specifies
        # the minimal significance necessary to consider the result.
        #
        # The following table can be used to choose <cut_off>
        # 95th percentile; 5% level; p < 0.05; critical value = 3.84
        # 99th percentile; 1% level; p < 0.01; critical value = 6.63
        # 99.9th percentile; 0.1% level; p < 0.001; critical value = 10.83
        # 99.99th percentile; 0.01% level; p < 0.0001; critical value = 15.13
        #
        # <sig_token> is statistical function used to calculate significance,
        # the possible arguments are 'LL' for dunning log-likelyhood and 'FE'
        # for fisher's exact test 
     
        if sig_token == 'FE': 
            sig_func = lambda x : 1.0 - fisher_exact(x)[1]
        else:
            sig_func = dunning_significance

        ref_text = read_all_corpus(corpus_ref)
        ref_len  = len(ref_text)
        text_len = len(text) 

        fd       = FreqDist([w.lower() for w in text])
        fd_ref   = FreqDist([w.lower() for w in ref_text])

        list_p_diff_pos = []
        list_p_diff_neg = []
        list_p_diff_unq = []
        for w in fd.keys():
                if fd_ref[w] != 0: 
                        p_diff_val = p_diff( (fd[w], text_len), (fd_ref[w],
                            ref_len))
                        if p_diff_val >=0:
                            list_p_diff_pos.append( (p_diff_val, w) )
                        else:
                            list_p_diff_neg.append( (p_diff_val, w) )
                else: 
                        list_p_diff_unq.append( (fd[w] ,w) ) 
        list_p_diff_pos = sorted(list_p_diff_pos, key = lambda x: x[0], reverse =
                        True)
        list_p_diff_neg = sorted(list_p_diff_neg, key = lambda x: -x[0], reverse =
                        True)
        list_p_diff_unq = sorted(list_p_diff_unq, key = lambda x: x[0], reverse =
                        True)

        output_pos = []
        output_neg = []
        output_unq = []
        i          = 0
        while i < len(list_p_diff_pos) and len(output_pos) < N:
                w   = list_p_diff_pos[i][1]
                sig = sig_func( [[fd[w], text_len], [fd_ref[w],
                        ref_len]])
                if sig > cut_off:
                        output_pos.append((w, list_p_diff_pos[i][0], sig))
                i += 1

        i = 0
        while i < len(list_p_diff_neg) and len(output_neg) < N:
                w   = list_p_diff_neg[i][1]
                sig = sig_func( [[fd[w], text_len], [fd_ref[w],
                        ref_len]])
                if sig > cut_off:
                        output_neg.append((w, list_p_diff_neg[i][0], sig))
                i += 1

        i = 0 
        while i < len(list_p_diff_unq) and len(output_unq) < N:
                w   = list_p_diff_unq[i][1]
                sig = sig_func( [[fd[w], text_len], [fd_ref[w],
                        ref_len]])
                if sig > cut_off:
                        output_unq.append((w, fd[w], sig))
                i += 1

        return (output_pos, output_neg, output_unq)


def print_N_keywords(text, corpus_ref, N, cut_off, sig_token):
        # A function which prints the results of the N_keywords function one
        # keyword per line
        l_pos, l_neg, l_unq = N_keywords(text, corpus_ref, N, cut_off, sig_token)
        print "\n%d POSITIVE significant keywords where found" % len(l_pos)
        for el in l_pos:
                print "Keyword: %s, %%DIFF: %0.2f, Significance: %0.5f" % el 

        print "\n%d NEGATIVE significant keywords where found" % len(l_neg)
        for el in l_neg:
                print "Keyword: %s, %%DIFF: %0.2f, Significance: %0.5f" % el 

        print "\n%d UNIQUE significant keywords where found" % len(l_unq)
        for el in l_unq:
                print "Keyword: %s, Frequency in the text: %i, Significance: %0.5f" % el 
 
########################################
## Collocations 
########################################
##
## We consider a collocation to be an above chance co-occurance of words,
## wherewords are considered to be co-occuring if they are at most <offset>
## words appart fromeach other.

def contingency_table(text, target_word, offset):
        # <text> is tokenized and lowercase 
        table  = defaultdict(lambda : [0, 0])
        length = len(text)
        nr_target = 0
        target_lower = target_word.lower()
        for i in range(offset, length - offset):
              # if i % 100000 == 0:
              #         print "%d of %d" % (i, length)
                low  = max(0, i - offset)
                high = min(length - 1, i + offset) 
                if text[i] == target_lower:
                        nr_target += 1
                        seen = {}
                        for j in range(low, i):
                                word = text[j]
                                if not word in seen:
                                        table[word][0] += 1
                                        seen[word] = True
                        for j in range(i + 1, high + 1):
                                word = text[j]
                                if not word  in seen:
                                        table[word][0] += 1
                                        seen[word] = True
                else:
                        seen = {}
                        for j in range(low, i):
                                word = text[j]
                                if not word in seen:
                                        table[word][1] += 1
                                        seen[word] = True
                        for j in range(i + 1, high + 1):
                                word = text[j]
                                if not word  in seen:
                                        table[word][1] += 1
                                        seen[word] = True
        return (table, nr_target) 


def collocations(text, target_word, offset, cut_off, sig_token='LL'):

        if sig_token == 'FE': 
            sig_func = lambda x : 1.0 - fisher_exact(x)[1]
        else:
            sig_func = dunning_significance

        text_lower = [w.lower() for w in text]
        table, nr_target = contingency_table(text_lower, target_word, offset)
        tot_len = len(text_lower)
        output = []
        for word, l in table.items():
            if float(l[0])/(l[0] + l[1]) > float(nr_target - l[0])/(tot_len - l[0] - l[1]):
                sig = sig_func([(l[0], l[0] + l[1]), (nr_target -
                        l[0], tot_len - l[0] - l[1])])
                if sig > cut_off:
                        output.append((word, sig))
        return sorted(output, key = itemgetter(1), reverse = True)


def N_collocations(text, target_word, offset, cut_off, N, sig_token='LL'):
        return collocations(text, target_word, offset, cut_off, sig_token)[:N]

def print_N_collocations(text, target_word, offset, cut_off, N, sig_token='LL'):
        for el in N_collocations(text, target_word, offset, cut_off, N, sig_token):
                print "Collocate: %s, Significance: %0.5f" % el 


def collocations_intersection(text, text_ref, target_word, offset, cut_off, sig_token='LL'):
        collocates     = set(collocations(text, target_word, offset, cut_off, sig_token)) 
        collocates_ref = collocations(text_ref, target_word, offset, cut_off, sig_token) 
        return [(val,sig,sig2) for val, sig in collocates for val2, sig2 in collocates_ref if val == val2]

def print_N_collocations_intersection(text, text_ref, target_word, offset, cut_off, N, sig_token='LL'):
        intersection = collocations_intersection(text, text_ref, target_word, offset, cut_off, sig_token) 
        if len(intersection) == 0:
                print "There are no collocates of %s in text which are also collocates in reference text" % target_word
        for el in intersection[:N]:
                print "Collocate: %s, Significanc in main text: %0.5f, Significance in reference text %0.5f" % el 

    
 
def N_collocations_with_ref(text, text_ref, target_word, offset, cut_off, N, sig_token='LL'):
        # The function finds <N> significant collocations in the text <text>
        # with respect to the reference corpus. The variable <cut_off>
        # specifies the minimal log-likehood necessary to consider the result
        # as significant.
        #
        # The following table can be used to choose <cut_off>
        # 95th percentile; 5% level; p < 0.05; critical value = 3.84
        # 99th percentile; 1% level; p < 0.01; critical value = 6.63
        # 99.9th percentile; 0.1% level; p < 0.001; critical value = 10.83
        # 99.99th percentile; 0.01% level; p < 0.0001; critical value = 15.13

        if sig_token == 'FE': 
            sig_func = lambda x : 1.0 - fisher_exact(x)[1]
        else:
            sig_func = dunning_significance

        text_lower = [w.lower() for w in text]
        contingency_tab, occurence_target = contingency_table(text_lower,
                                                            target_word, offset)

        text_ref_lower = [w.lower() for w in text_ref] 
        contingency_tab_ref, occurence_target_ref = contingency_table(text_ref_lower,
                                                                    target_word, offset)

        # First consider collocations in text and compute their diff values 
        tot_len = len(text_lower)
        collocations_text = []
        for word, l in contingency_tab.items():
            if (l[0] + l[1] > 0 and tot_len - l[0] - l[1] > 0 and
               float(l[0])/(l[0] + l[1]) > float(occurence_target - l[0])/(tot_len - l[0] - l[1])):
                sig = sig_func([(l[0], l[0] + l[1]), (occurence_target -
                        l[0], tot_len - l[0] - l[1])])
                if sig > cut_off:
                        collocations_text.append(word)

        list_p_diff_pos = []
        list_p_diff_neg = []
        list_p_diff_unq = []
        for w in collocations_text:
                if w in contingency_tab_ref and contingency_tab_ref[w][0] > 0:
                    p_diff_val = p_diff( (contingency_tab[w][0], occurence_target),
                                         (contingency_tab_ref[w][0], occurence_target_ref) )
                    if p_diff_val >=0:
                        list_p_diff_pos.append( (p_diff_val, w) )
                    else:
                        list_p_diff_neg.append( (p_diff_val, w) )
                else: # w does not appear in the vicinity of target_word in reference corpus
                        list_p_diff_unq.append( (contingency_tab[w][0], w) ) 

        list_p_diff_pos = sorted(list_p_diff_pos, key = lambda x: x[0], reverse =
                        True)
        list_p_diff_neg = sorted(list_p_diff_neg, key = lambda x: -x[0], reverse =
                        True)
        list_p_diff_unq = sorted(list_p_diff_unq, key = lambda x: x[0], reverse =
                        True)

        output_pos = []
        output_neg = []
        output_unq = []
        i          = 0
        while i < len(list_p_diff_pos) and len(output_pos) < N:
                w   = list_p_diff_pos[i][1]
                sig = sig_func([(contingency_tab[w][0], occurence_target),
                                            (contingency_tab_ref[w][0], occurence_target_ref)])
                if sig > cut_off:
                        output_pos.append((w, list_p_diff_pos[i][0], sig, contingency_tab[w][0]))
                i += 1

        i = 0
        while i < len(list_p_diff_neg) and len(output_neg) < N:
                w   = list_p_diff_neg[i][1]
                sig = sig_func([(contingency_tab[w][0], occurence_target),
                                            (contingency_tab_ref[w][0], occurence_target_ref)])
                if sig > cut_off:
                        output_neg.append((w, list_p_diff_neg[i][0], sig, contingency_tab[w][0]))
                i += 1

        i = 0 
        while i < len(list_p_diff_unq) and len(output_unq) < N:
                w   = list_p_diff_unq[i][1]
                sig = sig_func([(contingency_tab[w][0], occurence_target),
                                            (contingency_tab_ref[w][0], occurence_target_ref)])
                if sig > cut_off:
                        output_unq.append((w, contingency_tab[w][0], sig))
                i += 1

        # Now do the same sort of analysis for the collocations text_ref
        tot_len_ref = len(text_ref_lower)
        collocations_text_ref = []
        for word, l in contingency_tab_ref.items():
            if (l[0] + l[1] > 0 and tot_len_ref - l[0] - l[1] > 0 and
               float(l[0])/(l[0] + l[1]) > float(occurence_target_ref - l[0])/(tot_len_ref - l[0] - l[1])):
                sig = sig_func([(l[0], l[0] + l[1]), (occurence_target_ref -
                        l[0], tot_len_ref - l[0] - l[1])])
                if sig > cut_off:
                        collocations_text_ref.append(word)

        list_p_diff_pos = []
        list_p_diff_neg = []
        list_p_diff_unq = []
        for w in collocations_text_ref:
                if w in contingency_tab and contingency_tab[w][0] > 0:
                    p_diff_val = p_diff( (contingency_tab[w][0], occurence_target),
                                         (contingency_tab_ref[w][0], occurence_target_ref) )
                    if p_diff_val >=0:
                        list_p_diff_pos.append( (p_diff_val, w) )
                    else:
                        list_p_diff_neg.append( (p_diff_val, w) )
                else: # w does not appear in the vicinity of target_word in reference corpus
                        list_p_diff_unq.append( (contingency_tab_ref[w][0], w) ) 

        list_p_diff_pos = sorted(list_p_diff_pos, key = lambda x: x[0], reverse =
                        True)
        list_p_diff_neg = sorted(list_p_diff_neg, key = lambda x: -x[0], reverse =
                        True)
        list_p_diff_unq = sorted(list_p_diff_unq, key = lambda x: x[0], reverse =
                        True)

        output_pos_ref = []
        output_neg_ref = []
        output_unq_ref = []
        i              = 0
        while i < len(list_p_diff_pos) and len(output_pos_ref) < N:
                w   = list_p_diff_pos[i][1]
                sig = sig_func([(contingency_tab[w][0], occurence_target),
                                            (contingency_tab_ref[w][0], occurence_target_ref)])
                if sig > cut_off:
                        output_pos_ref.append((w, list_p_diff_pos[i][0], sig, contingency_tab_ref[w][0]))
                i += 1

        i = 0
        while i < len(list_p_diff_neg) and len(output_neg_ref) < N:
                w   = list_p_diff_neg[i][1]
                sig = sig_func([(contingency_tab[w][0], occurence_target),
                                            (contingency_tab_ref[w][0], occurence_target_ref)])
                if sig > cut_off:
                        output_neg_ref.append((w, list_p_diff_neg[i][0], sig, contingency_tab_ref[w][0]))
                i += 1

        i = 0 
        while i < len(list_p_diff_unq) and len(output_unq_ref) < N:
                w   = list_p_diff_unq[i][1]
                sig = sig_func([(contingency_tab[w][0], occurence_target),
                                            (contingency_tab_ref[w][0], occurence_target_ref)])
                if sig > cut_off:
                        output_unq_ref.append((w, contingency_tab_ref[w][0], sig))
                i += 1


        return ((output_pos, output_neg, output_unq), (output_pos_ref, output_neg_ref, output_unq_ref))


def print_N_collocations_with_ref(text, text_ref, target_word, offset, cut_off, N, sig_token='LL'):
        (l_pos, l_neg, l_unq), (ll_pos, ll_neg, ll_unq) = N_collocations_with_ref(
        text, text_ref, target_word, offset, cut_off, N, sig_token)
       
        print "Considering the collocates of %s in the text\n\n\n" % target_word 

        print "%d POSITIVE significant collocates where found" % len(l_pos)
        for el in l_pos:
                print "Collocate: %s, %%DIFF: %0.2f, Significance: %0.5f, Frequency: %i" % el 

        print "\n%d NEGATIVE significant collocates where found" % len(l_neg)
        for el in l_neg:
                print "Collocate: %s, %%DIFF: %0.2f, Significance: %0.5f, Frequency: %i" % el 

        print "\n%d UNIQUE significant collocates where found" % len(l_unq)
        for el in l_unq:
                print "Collocate: %s, Frequency in the text: %i, Significance: %0.5f" % el 

        print "\n\n\nConsidering the collocates of %s in the reference text\n\n\n" % target_word 

        print "%d POSITIVE significant collocates where found" % len(ll_pos)
        for el in ll_pos:
                print "Collocate: %s, %%DIFF: %0.2f, Significance: %0.5f, Frequency: %i" % el 

        print "\n%d NEGATIVE significant collocates where found" % len(ll_neg)
        for el in ll_neg:
                print "Collocate: %s, %%DIFF: %0.2f, Significance: %0.5f, Frequency: %i" % el 

        print "\n%d UNIQUE significant collocates where found" % len(ll_unq)
        for el in ll_unq:
                print "Collocate: %s, Frequency in the text: %i, Significance: %0.5f" % el 
 
def print_concordance_lines(text, node, collocate, offset_col, offset_print):
        L = len(text)
        for i in range(L):
                if text[i] == node:
                        mi = max(0, i - offset_col)
                        ma = min(L, i + offset_col + 1)
                        for j in range(mi, ma):
                                if text[j] == collocate:
                                        print " ".join(text[max(0, i - offset_print):
                                                            min(L, i + offset_print + 1)])
                                        break

def intersection_N_most_popular(text, text2, N, exclude):
        set_exclude = set(exclude)
        M = len(text)
        M2= len(text2)
#        tokens = [w.lower() for w in text]
#        tokens2 = [w.lower() for w in text2]
        fd = FreqDist(text)
        new = [] 
        sort = sorted(fd.items(), key = itemgetter(1), reverse = True)
        j = 0
        while len(new) < N and j < len(sort):
                if not sort[j][0] in set_exclude:
                        new.append(sort[j][0])
                j += 1
        fd2 = FreqDist(text2)
        new2 = [] 
        sort = sorted(fd2.items(), key = itemgetter(1), reverse = True)
        j = 0
        while len(new2) < N and j < len(sort):
                if not sort[j][0] in set_exclude:
                        new2.append(sort[j][0])
                j += 1
        total = 0
        for word in new: 
                if word in new2:
                        print word, 1.0*fd[word]/M, 1.0*fd2[word]/M2
                        total += 1
        print "%i words in the intersection" % total

########################################
## Boring words
########################################

WORDS_TO_EXCLUDE = [u'm\xe1s', u'est\xe1n', u'est\xe1', u'tambi\xe9n',
u's\xed', u'qu\xe9', 'yo', 'ya', 'sin', 'esta', 'van', 'este', 'estamos',
'nuestros', 'otra', 'donde', 'todo', '.',',', 'de', 'la', 'y', 'a', 'en',
'que', 'se', 'ya', 'le', 'me', 'ni', '-', '...', 'pues', '?', 'ha', 'de',
'los', 'las', 'del', 'por', ':', '"', 'el', 'no', 'es', 'lo', 'para', 'con',
'su', 'para', 'porque', 'como', 'un', '(', ')', '@', 'o', 'nos', 'al',
'entonces', 'su', 'una', 'pero', 'si', 'son', 'eso', 'les', 'hay', 'sus',
'nuestro', 'nosotros', u'\u201c', u'\u201d', u'\xbf', u'\u2026', u'\u2013',
u'\u201d,', u'\u201d.', u'),', u';', u'.-', u'\u2014', u').', "'", u'\u2019',
u'\xad', u'\xa1', u'\xab', u's', u'e', u'l', u'!', u'\u2014,',u'\u201c,',
u'\u201c.', u'\u2212', u'\xa1', u'\xab', u's', u'e', u'l', u'!', u'cuando',
u'va', u'nuestra', u'as\xed', u'ser', u'hacer', u'tiene', u'ese', u'desde',
u'ellos', u'sobre', u'ellos', u'muy', u'entre', u'sea', u'han', u'sino',
u'esto', u'all\xe1', u'ustedes', u'/', u'ah\xe1', u'aqu\xe1']
