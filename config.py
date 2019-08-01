import numpy as np
from easydict import EasyDict as edict
import re

__C = edict()
cfg = __C

# Path to concept_vqa
__C.ROOT_DIR = 'concept_vqa' # '/home/lyt/code/vqa-concept'

# Path to data
__C.DATA_DIR = '{}/dataTVQA'.format(__C.ROOT_DIR)

# Path to image feature
__C.FEA_DIR = '{}/image-feature/bottomup'.format(__C.DATA_DIR)

# Path to ocr feature
__C.OCR_DIR = '{}/image-feature/ocr'.format(__C.DATA_DIR)

# Path to TextVQA
__C.TVQA_DIR = 'TextVQA' # '/home/data/lyt/TextVQA'

# Path to TextVQA images
__C.IMAGE_TRAIN_DIR = '{}/train_images'.format(__C.TVQA_DIR)
__C.IMAGE_TEST_DIR = '{}/test_images'.format(__C.TVQA_DIR)

# Path to vqa tools
__C.VQA_DIR = 'vqa-tools'


# ============= image_id:ocr =============
# input: TextVQA original jsdata['data']
def get_im_ocr(data):
    im_ocr = {}
    for pair in data:
        im_id = pair['image_id']
        if im_id not in im_ocr:
            im_ocr[im_id] = {'image_classes': pair['image_classes'],
                             'image_width': pair['image_width'],
                             'image_height': pair['image_height'],
                             'ocr_tokens': pair['ocr_tokens'],
                             'ocr_info': pair['ocr_info']}
    return im_ocr

# ============= word embedding =============
# Load word2vec embedding
def get_emb(emb_file, emb_size):
    with open(emb_file) as f:
        raw = f.read().splitlines()
    word_vec = [l.split(' ', 1) for l in raw]
    vocab, vecs_txt = zip(*word_vec)
    vecs = np.fromstring(' '.join(vecs_txt), dtype='float32', sep=' ')
    vecs = vecs.reshape(-1, emb_size)
    emb_dict = dict(zip(vocab, vecs))
    return emb_dict, np.mean(vecs), np.std(vecs)

# Load word2vec vocab
def load_vocab(emb_file):
    with open(emb_file) as f:
        raw = f.read().splitlines()
    vocab = [l.split(' ', 1)[0] for l in raw]
    return vocab

# Load word2vec vocab and embedding
def load_vocab_emb(emb_file, emb_size):
    with open(emb_file) as f:
        raw = f.read().splitlines()
    word_vec = [l.split(' ', 1) for l in raw]
    vocab, vecs_txt = zip(*word_vec)
    vecs = np.fromstring(' '.join(vecs_txt), dtype='float32', sep=' ')
    vecs = vecs.reshape(-1, emb_size)
    return vocab, vecs

# Write word2vec embedding
def write_vocab_emb(emb_file, vocab, emb):
    emb = emb.tolist()
    emb = [['{:f}'.format(xx) for xx in x] for x in emb]
    vocab = [[v] for v in vocab]
    y = list(zip(vocab,emb))
    z = [' '.join(yy[0]+yy[1]) for yy in y]
    with open(emb_file, 'w') as f:
        f.write('\n'.join(z))


# ============= norm text =============
# borrow from vqaEval.py
m_contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've",
    "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't",
    "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
    "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've",
    "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's",
    "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "twas": "'twas",
    "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
    "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've"}
m_manual_map = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
                'four': '4', 'five': '5', 'six': '6', 'seven': '7',
                'eight': '8', 'nine': '9', 'ten': '10'}
m_articles = ['a', 'an', 'the']
m_period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
m_comma_strip = re.compile("(\d)(\,)(\d)")
m_punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+',
           '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']

def process_punct(in_text):
    out_text = in_text
    for p in m_punct:
        if (p + ' ' in in_text or ' ' + p in in_text
                or re.search(m_comma_strip, in_text) != None):
            out_text = out_text.replace(p, '')
        else:
            out_text = out_text.replace(p, ' ')
    out_text = m_period_strip.sub("", out_text, re.UNICODE)
    return out_text

def process_digit_article(in_text):
    out_text = []
    for word in in_text.lower().split():
        if word not in m_articles:
            word = m_manual_map.setdefault(word, word)
            word = m_contractions.setdefault(word, word)
            out_text.append(word)
    return ' '.join(out_text)

def normText(text):
    text = process_punct(text)
    text = process_digit_article(text)
    return text

def stripLower(text):
    text = text.replace('\n', ' ').replace('\t', ' ').strip().lower()
    return text
