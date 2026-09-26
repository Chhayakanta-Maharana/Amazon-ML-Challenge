import re
from collections import defaultdict
from preprocessing import clean_name, clean_address, clean_country

# Words that appear too frequently to be useful single-token blocking keys
GENERIC_STOPWORDS = {
    "the", "and", "for", "with", "from", "inc", "corp", "ltd", "pvt", "llc",
    "llp", "sa", "sarl", "co", "company", "services", "service", "enterprises",
    "enterprise", "solutions", "tech", "technologies", "group", "holdings",
    "associates", "consulting", "international", "global", "national", "center",
    "centre", "store", "shop", "mart", "care", "india", "us", "usa", "france"
}

class Blocker:
    def __init__(self, max_candidates_per_entity=60):
        self.max_candidates = max_candidates_per_entity
        # Inverted index: country -> key -> list of entity_ids
        self.name_token_index = defaultdict(lambda: defaultdict(list))
        self.name_prefix_index = defaultdict(lambda: defaultdict(list))
        self.sorted_tokens_index = defaultdict(lambda: defaultdict(list))
        self.address_key_index = defaultdict(lambda: defaultdict(list))
        
        # Stored records for fast retrieval
        self.records = {}
        
    def index_target_records(self, df_records):
        """Index Source 2 and Source 3 records."""
        for _, row in df_records.iterrows():
            eid = row['entity_id']
            country = clean_country(row['country'])
            n_data = clean_name(row['business_name'])
            a_data = clean_address(row['business_address'])
            
            self.records[eid] = (country, n_data, a_data)
            
            # 1. Name Prefix Key (e.g. first 5 chars of alnum)
            if len(n_data['alnum']) >= 4:
                prefix_key = n_data['alnum'][:5]
                self.name_prefix_index[country][prefix_key].append(eid)
                
            # 2. Sorted First 2 Tokens Key
            tokens_filtered = [t for t in n_data['tokens'] if t not in GENERIC_STOPWORDS and len(t) >= 3]
            if len(tokens_filtered) >= 2:
                sort_pair_key = "_".join(sorted(tokens_filtered[:2]))
                self.sorted_tokens_index[country][sort_pair_key].append(eid)
                
            # 3. Significant Name Tokens (limit inverted index fanout)
            for token in tokens_filtered:
                if len(token) >= 3:
                    self.name_token_index[country][token].append(eid)
                    
            # 4. Address Numeric + Distinct Token Key
            if a_data['num_tokens'] and a_data['tokens']:
                addr_words = [t for t in a_data['tokens'] if t not in GENERIC_STOPWORDS and len(t) >= 4 and not t.isdigit()]
                if addr_words:
                    for num in list(a_data['num_tokens'])[:2]:
                        for word in addr_words[:2]:
                            addr_key = f"{num}_{word}"
                            self.address_key_index[country][addr_key].append(eid)

    def generate_candidates_for_entity(self, s1_id, raw_name, raw_address, raw_country):
        """Generate candidates for a single Source 1 entity across all channels."""
        country = clean_country(raw_country)
        n_data = clean_name(raw_name)
        a_data = clean_address(raw_address)
        
        candidates = set()
        channel_hits = defaultdict(set)
        
        # Channel 1: Name Prefix
        if len(n_data['alnum']) >= 4:
            prefix_key = n_data['alnum'][:5]
            hits = self.name_prefix_index[country].get(prefix_key, [])
            if len(hits) <= 200:
                for h in hits:
                    candidates.add(h)
                    channel_hits[h].add("name_prefix")
                    
        # Channel 2: Sorted First 2 Tokens Key
        tokens_filtered = [t for t in n_data['tokens'] if t not in GENERIC_STOPWORDS and len(t) >= 3]
        if len(tokens_filtered) >= 2:
            sort_pair_key = "_".join(sorted(tokens_filtered[:2]))
            hits = self.sorted_tokens_index[country].get(sort_pair_key, [])
            if len(hits) <= 200:
                for h in hits:
                    candidates.add(h)
                    channel_hits[h].add("sorted_pair")
                    
        # Channel 3: Significant Name Tokens
        for token in tokens_filtered:
            hits = self.name_token_index[country].get(token, [])
            if 0 < len(hits) <= 150: # Avoid high-fanout generic collisions
                for h in hits:
                    candidates.add(h)
                    channel_hits[h].add("name_token")
                    
        # Channel 4: Address Numeric + Street Key
        if a_data['num_tokens'] and a_data['tokens']:
            addr_words = [t for t in a_data['tokens'] if t not in GENERIC_STOPWORDS and len(t) >= 4 and not t.isdigit()]
            if addr_words:
                for num in list(a_data['num_tokens'])[:2]:
                    for word in addr_words[:2]:
                        addr_key = f"{num}_{word}"
                        hits = self.address_key_index[country].get(addr_key, [])
                        if 0 < len(hits) <= 150:
                            for h in hits:
                                candidates.add(h)
                                channel_hits[h].add("addr_num_word")
                                
        # If candidate pool is too large, prioritize candidates that hit multiple channels
        if len(candidates) > self.max_candidates:
            ranked_cands = sorted(candidates, key=lambda c: len(channel_hits[c]), reverse=True)
            candidates = set(ranked_cands[:self.max_candidates])
            
        return list(candidates), channel_hits
