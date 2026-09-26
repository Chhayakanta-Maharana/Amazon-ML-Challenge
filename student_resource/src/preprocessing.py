import re
import unicodedata

# Common legal and generic entity terms
LEGAL_SUFFIX_MAP = {
    r'\b(private limited|pvt ltd|pvt\. ltd\.|pvt limited|p\. ltd|p\.ltd)\b': ' pvt ltd ',
    r'\b(limited|ltd|ltd\.)\b': ' ltd ',
    r'\b(incorporated|inc|inc\.)\b': ' inc ',
    r'\b(corporation|corp|corp\.)\b': ' corp ',
    r'\b(limited liability company|llc|l\.l\.c\.)\b': ' llc ',
    r'\b(limited liability partnership|llp|l\.l\.p\.)\b': ' llp ',
    r'\b(societe a responsabilite limitee|sarl|s\.a\.r\.l\.)\b': ' sarl ',
    r'\b(societe anonyme|sa|s\.a\.)\b': ' sa ',
    r'\b(society|soc|soc\.)\b': ' soc ',
    r'\b(association|assoc|assoc\.)\b': ' assoc ',
    r'\b(company|co|co\.)\b': ' co ',
}

# Common address abbreviations
ADDRESS_ABBR_MAP = {
    r'\b(avenue|ave|ave\.)\b': ' ave ',
    r'\b(street|str|st|st\.)\b': ' st ',
    r'\b(road|rd|rd\.)\b': ' rd ',
    r'\b(drive|dr|dr\.)\b': ' dr ',
    r'\b(boulevard|blvd|blvd\.)\b': ' blvd ',
    r'\b(lane|ln|ln\.)\b': ' ln ',
    r'\b(court|ct|ct\.)\b': ' ct ',
    r'\b(highway|hwy|hwy\.)\b': ' hwy ',
    r'\b(parkway|pkwy|pkwy\.)\b': ' pkwy ',
    r'\b(apartment|apt|apt\.)\b': ' apt ',
    r'\b(suite|ste|ste\.)\b': ' ste ',
    r'\b(floor|fl|flr)\b': ' fl ',
    r'\b(building|bldg|bldg\.)\b': ' bldg ',
    r'\b(north|n|n\.)\b': ' n ',
    r'\b(south|s|s\.)\b': ' s ',
    r'\b(east|e|e\.)\b': ' e ',
    r'\b(west|w|w\.)\b': ' w ',
    r'\b(near|nr|nr\.)\b': ' near ',
    r'\b(opposite|opp|opp\.)\b': ' opp ',
}

# Compile regex patterns for high-speed execution
LEGAL_COMPILED = [(re.compile(p, re.IGNORECASE), repl) for p, repl in LEGAL_SUFFIX_MAP.items()]
ADDR_COMPILED = [(re.compile(p, re.IGNORECASE), repl) for p, repl in ADDRESS_ABBR_MAP.items()]

RE_NON_ALNUM = re.compile(r'[^\w\s]', re.UNICODE)
RE_SPACES = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\b\d+\b')

def normalize_unicode(text):
    if not text or not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFKD', text)

def clean_name(raw_name):
    """Generates cleaned representations of a business name."""
    if not raw_name or not isinstance(raw_name, str):
        return {
            "raw": "",
            "clean": "",
            "alnum": "",
            "tokens": [],
            "sorted_tokens": "",
        }
    
    text = normalize_unicode(raw_name).lower()
    
    # Standardize legal suffixes
    for pattern, repl in LEGAL_COMPILED:
        text = pattern.sub(repl, text)
        
    # Standardize ampersands
    text = text.replace('&', ' and ').replace('+', ' plus ')
    
    # Strip symbols/punctuation
    clean_text = RE_NON_ALNUM.sub(' ', text)
    clean_text = RE_SPACES.sub(' ', clean_text).strip()
    
    tokens = [t for t in clean_text.split() if t]
    sorted_tokens = " ".join(sorted(tokens))
    alnum = "".join(tokens)
    
    return {
        "raw": raw_name,
        "clean": clean_text,
        "alnum": alnum,
        "tokens": tokens,
        "sorted_tokens": sorted_tokens,
    }

def clean_address(raw_address):
    """Generates cleaned representations of a business address."""
    if not raw_address or not isinstance(raw_address, str):
        return {
            "raw": "",
            "clean": "",
            "tokens": [],
            "sorted_tokens": "",
            "num_tokens": set(),
        }
    
    text = normalize_unicode(raw_address).lower()
    
    # Standardize address terms
    for pattern, repl in ADDR_COMPILED:
        text = pattern.sub(repl, text)
        
    text = text.replace('&', ' and ').replace('/', ' ').replace('#', ' ').replace('-', ' ')
    clean_text = RE_NON_ALNUM.sub(' ', text)
    clean_text = RE_SPACES.sub(' ', clean_text).strip()
    
    tokens = [t for t in clean_text.split() if t]
    sorted_tokens = " ".join(sorted(tokens))
    num_tokens = set(RE_NUMBERS.findall(clean_text))
    
    return {
        "raw": raw_address,
        "clean": clean_text,
        "tokens": tokens,
        "sorted_tokens": sorted_tokens,
        "num_tokens": num_tokens,
    }

def clean_country(raw_country):
    """Clean country string generically without hardcoding allowed countries."""
    if not raw_country or not isinstance(raw_country, str):
        return "unknown"
    return normalize_unicode(raw_country).strip().upper()
