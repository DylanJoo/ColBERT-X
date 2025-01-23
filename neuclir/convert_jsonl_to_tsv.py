import json

# for neuclir-csl corpus
# with open('/home/jju/datasets/neuclir-csl/csl.jsonl', 'r') as f, \
#      open('/home/jju/datasets/neuclir-csl/csl.tsv', 'w') as w, \
#      open('/home/jju/datasets/neuclir-csl/csl_mapping.tsv', 'w') as wmap:
#
#     for i, line in enumerate(f):
#         data = json.loads(line)
#         docid = data['doc_id']
#         doctext = data['title'] + " " + data['abstract']
#
#         w.write(f"{i}\t{doctext}\n")
#         wmap.write(f"{i}\t{docid}\n")

# for neuclir-csl topics
with open('neuclir-2023-technical_topics.0719.jsonl', 'r') as f, \
     open('neuclir-2023-technical_topics.0719.tsv', 'w') as w:
    for line in f:
        data = json.loads(line)
        qid = data['topic_id']
        qtext = data['topics'][0]['topic_title'] + " " + data['topics'][0]['topic_description'] 
        w.write(f"{qid}\t{qtext}\n")

