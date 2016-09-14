from itertools import chain
import re
from difflib import SequenceMatcher

def get_as_dict(x):
    """Return an object as a dictionary of its attributes"""
    if isinstance(x, dict):
        return x
    else:
        try:
            return x._asdict()
        except AttributeError:
            return x.__dict__

def sort_X_on_Y(X, Y):
    return [x for (y,x) in sorted(zip(Y,X), key=lambda t : t[0])]

def corenlp_cleaner(words):
    d = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{',
       '-RSB-': ']', '-LSB-': '['}
    return map(lambda w: d[w] if w in d else w, words)

def split_html_attrs(attrs):
    """
    Given an iterable object of (attr, values) pairs, returns a list of separated
    "attr=value" strings
    """
    html_attrs = []
    for a in attrs:
        attr = a[0]
        values = [v.split(';') for v in a[1]] if isinstance(a[1],list) else [a[1].split(';')]
        for i in range(len(values)):
            while isinstance(values[i], list):
                values[i] = values[i][0]
        html_attrs += ["=".join([attr,val]) for val in values]
    return html_attrs

def slice_into_ngrams(tokens, n_max=3, n_min=1, delim='_'):
    N = len(tokens)
    for root in range(N):
        for n in range(max(0,n_min-1), min(n_max, N - root)):
            yield delim.join(tokens[root:root+n+1])

def expand_implicit_text(text):
    """
    Given a string, generates strings that are potentially implied by
    the original text. Two main operations are performed:
        1. Expanding ranges (X to Y; X ~ Y; X -- Y)
        2. Expanding suffixes (123X/Y/Z; 123X, Y, Z)
    If no implicit terms are found, yields just the original string.
    To get the correct output from complex strings, this function should be fed
    many Ngrams from a particular phrase.
    """
    DEBUG = False # Set to True to see intermediate values printed out.

    def atoi(num_str):
        '''
        Helper function which converts a string to an integer, or returns None.
        '''
        try:
            return int(num_str)
        except:
            pass
        return None

    ### Regex Patterns compile only once per function call.
    # This range pattern will find text that "looks like" a range.
    range_pattern = re.compile(ur'^(?P<start>[\w\/]+)(?:\s*(\.{3,}|\~|\-+|to|thru|through|\u2013+|\u2014+|\u2012+|\u2212+)\s*)(?P<end>[\w\/]+)$', re.IGNORECASE | re.UNICODE)
    suffix_pattern = re.compile(ur'(?P<spacer>(?:,|\/)\s*)(?P<suffix>[\w\-]+)')
    base_pattern = re.compile(ur'(?P<base>[\w\-]+)(?P<spacer>(?:,|\/)\s*)?(?P<suffix>[\w\-]+)?')

    if DEBUG: print "[debug] Text: " + text
    inferred_texts = set()
    final_set = set()

    ### Step 1: Search and expand ranges
    m = re.search(range_pattern, text)
    if m:
        start = m.group("start")
        end = m.group("end")
        start_diff = ""
        end_diff = ""
        if DEBUG: print "[debug]   Start: %s \t End: %s" % (start, end)

        # Use difflib to find difference. We are interested in 'replace' only
        seqm = SequenceMatcher(None, start, end).get_opcodes();
        for opcode, a0, a1, b0, b1 in seqm:
            if opcode == 'equal':
                continue
            elif opcode == 'insert':
                break
            elif opcode == 'delete':
                break
            elif opcode == 'replace':
                # NOTE: Potential bug if there is more than 1 replace
                start_diff = start[a0:a1]
                end_diff = end[b0:b1]
            else:
                raise RuntimeError, "[ERROR] unexpected opcode"


        if DEBUG: print "[debug]   start_diff: %s \t end_diff: %s" % (start_diff, end_diff)

        # Check Numbers
        if atoi(start_diff) and atoi(end_diff):
            if DEBUG: print "[debug]   Enumerate %d to %d" % (atoi(start_diff), atoi(end_diff))
            # generate a list of the numbers plugged in
            number_range = range(atoi(start_diff), atoi(end_diff) + 1)
            for number in number_range:
                new_text = start.replace(start_diff,str(number))
                # Produce the strings with the enumerated ranges
                inferred_texts.add(new_text)

        # Second, check for single-letter enumeration
        if len(start_diff) == 1 and len(end_diff) == 1:
            if start_diff.isalpha() and end_diff.isalpha():
                def char_range(a, b):
                    '''
                    Generates the characters from a to b inclusive.
                    '''
                    for c in xrange(ord(a), ord(b)+1):
                        yield chr(c)

                if DEBUG: print "[debug]   Enumerate %s to %s" % (start_diff, end_diff)
                letter_range = char_range(start_diff, end_diff)
                for letter in letter_range:
                    new_text = start.replace(start_diff,letter)
                    # Produce the strings with the enumerated ranges
                    inferred_texts.add(new_text)
    else: inferred_texts.add(text)
    if DEBUG: print "[debug]   Inferred Text: \n  " + str(sorted(inferred_texts))

    ### Step 2: Expand suffixes for each of the inferred phrases
    # NOTE: this only does the simple case of replacing same-length suffixes.
    # we do not handle cases like "BC546A/B/XYZ/QR"
    for text in inferred_texts:
        first_match = re.search(base_pattern,text)
        if first_match:
            base = re.search(base_pattern,text).group("base");
            final_set.add(base) # add the base (multiple times, but set handles that)
            if (first_match.group("suffix")):
                all_suffix_lengths = set()
                # This is a bit inefficient but this first pass just is here
                # to make sure that the suffixes are the same length
                for m in re.finditer(suffix_pattern, text):
                    suffix = m.group("suffix")
                    suffix_len = len(suffix)
                    all_suffix_lengths.add(suffix_len)
                if len(all_suffix_lengths) == 1:
                    for m in re.finditer(suffix_pattern, text):
                        spacer = m.group("spacer")
                        suffix = m.group("suffix")
                        suffix_len = len(suffix)
                        trimmed = base[:-suffix_len]
                        final_set.add(trimmed+suffix)
    if DEBUG: print "[debug]   Final Set: " + str(sorted(final_set))

    # Yield only the unique values
    for inferred_texts in final_set:
        yield inferred_texts

    # NOTE: We make a few assumptions (e.g. suffixes must be same length), but
    # one important unstated assumption is that if there is a single suffix,
    # (e.g. BC546A/B), the single suffix will be swapped in no matter what.
    # In this example, it works. But if we had "ABCD/EFG" we would get "ABCD,AEFG"
    # Check out UtilsTests.py to see more of our assumptions capture as test
    # cases.
