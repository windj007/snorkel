import os, sys, unittest
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from snorkel.parser import CorpusParser, HTMLParser, OmniParser
from snorkel.matchers import DictionaryMatch
from snorkel.candidates import TableNgrams, EntityExtractor
from snorkel.features import TableNgramFeaturizer
import cPickle
from time import sleep

ROOT = os.environ['SNORKELHOME']

class TestTables(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.context_parser = OmniParser()
        cls.base = ROOT + '/test/data/table_test/'
        cls.filename = 'diseases.xhtml'

    @classmethod
    def tearDownClass(cls):
        sleep(1)
        cls.context_parser.corenlp_handler._kill_pserver()

    def test_parsing(self):
        doc_parser = HTMLParser(path=self.base + self.filename)
        cp = CorpusParser(doc_parser, self.context_parser)
        corpus = cp.parse_corpus(name='Test Corpus')
        
        self.assertEqual(len(corpus.documents), 1)
        self.assertEqual(len(corpus.documents[0].tables), 2)
        self.assertEqual(len(corpus.documents[0].cells), 24)
        self.assertEqual(len(corpus.documents[0].phrases), 34)
        self.assertEqual(len(corpus.documents[0].tables[0].phrases), 16)
        self.assertEqual(corpus.documents[0].tables[0].cells[0].phrases[0].text, "Disease")

        # NOTE: Uncomment to save new pickle for downstream testing
        # with open(self.base + 'corpus.pkl', 'wb') as f:
            # cPickle.dump(corpus, f)

    def test_extraction(self):
        with open(self.base + 'corpus.pkl', 'rb') as f:
            corpus = cPickle.load(f)

        diseases = ['coughs','colds','brain cancer','brain','cancer','common',
                'ailments','disease','location','polio','plague','scurvy',
                'infectious diseases','infectious','diseases','problem','arthritis',
                'fever','hypochondria','pneumonia']
        table_ngrams = TableNgrams(n_max=3)
        disease_matcher = DictionaryMatch(d=diseases)
        ce = EntityExtractor(table_ngrams, disease_matcher)
        candidates = ce.extract(corpus.get_phrases(), name='all')
    
        self.assertEqual(len(candidates), 20)
        self.assertEqual(candidates[0].get_span(), 'coughs')
        
        # NOTE: Uncomment to save new pickle for downstream testing
        # with open(self.base + 'candidates.pkl', 'wb') as f:
        #     cPickle.dump(candidates, f)

    def test_featurization(self):
        with open(self.base + 'candidates.pkl', 'rb') as f:
            candidates = cPickle.load(f)

        featurizer = TableNgramFeaturizer()
        featurizer.fit_transform(candidates)        
        
        self.assertEqual(featurizer.get_features_by_candidate(
            candidates[0])[0],'DDLIB_WORD_SEQ_[coughs]')
        self.assertEqual(featurizer.get_features_by_candidate(
            candidates[0])[-1],'TABLE_HTML_ANC_ATTR_xmlns')
        

if __name__ == '__main__':
    unittest.main()
