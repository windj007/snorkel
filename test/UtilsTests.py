import os, sys, unittest
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from snorkel.utils import expand_implicit_text # add more here for each function
from time import sleep

ROOT = os.environ['SNORKELHOME']

class TestImplicitExpansion(unittest.TestCase):

    def test_single_word(self):
        phrase = "ABC"
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 1)
        self.assertEqual(phrase, lst[0])

    def test_empty_string(self):
        phrase = ""
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 0) # nothing is returned

    def test_letter_range_and_suffix(self):
        # Test '...' range + suffix combination
        phrase = "BC546A/B/C...BC550A/B/C"
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 15)

        # Test 'to' range + suffix combination
        phrase = "BC546A/B/C to BC550A/B/C"
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 15)

        # Test single letter range expansion
        phrase = "BC546A~BC546E"
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 5)
        phrase = "BC54A6~BC54E6" # test when letter is in middle of text
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 5)

        # Test simple number expansion
        phrase = "BC546/550/543"
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 3)

        # Test different length suffixes. Only return base.
        phrase = "BC547A, BC5XB, C"
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 1)
        self.assertEqual("BC547A", lst[0])

        # NOTE: One document has a phrase like this. Even though it seems
        # incorrect to a human, our the behavior of the function seems
        # reasonable. We can look at improving this case later.
        phrase = "BC182,A,B"
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 3)
        self.assertTrue("BC182" in lst)
        self.assertTrue("BC18A" in lst) # still just swap suffix in
        self.assertTrue("BC18B" in lst)

        phrase = "BC547A/BC546B"
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 2)
        self.assertTrue("BC547A" in lst)
        self.assertTrue("BC546B" in lst)

        phrase = "A to C"
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 3)
        self.assertTrue("A" in lst)
        self.assertTrue("B" in lst)
        self.assertTrue("C" in lst)

    def test_number_suffix(self):
        phrase = "BC546-16/-25/-40"
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 3)
        self.assertTrue("BC546-16" in lst)
        self.assertTrue("BC546-25" in lst)
        self.assertTrue("BC546-40" in lst)

    @unittest.skip("No evidence of this happening in the wild, but documenting it as a potential case.")
    def test_complicated_dash_pattern(self):
        # NOTE: not really sure what to think about something like this. I
        # have no evidence of this happening in the wild. This is an expected
        # failure we can address later.
        phrase = "BC546-16/-25/-40 ~ BC550-16/-25/-40"
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 15)

    def test_single_suffix(self):
        # NOTE: This seems like reasonable behavior. If only one suffix is
        # present, we swap it in no matter what.
        phrase = "ABCD/EFG"
        lst = list(expand_implicit_text(phrase))
        self.assertEqual(len(lst), 2)
        self.assertTrue("ABCD" in lst)
        self.assertTrue("AEFG" in lst)


if __name__ == '__main__':
    unittest.main()
