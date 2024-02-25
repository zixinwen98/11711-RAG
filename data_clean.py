from utils import * 

def data_clean(s):
    s = s.strip()
    s = ' '.join(s.split())
    return s 

clean_path = ['commencement_schedule.json',
              'fact_sheet.json', 'handbook_texts.json', 'kiltie_band_fact.json',
              'LTI_Programs.json', 'scotty_fact.json', 'tartan_fact.json', 'buggy_history.json', 'carnival_events.json',
              'cmu_history.json']

for path in clean_path:
    f = jload('/zfsauton2/home/yifuc/11711-RAG/data/cmu/' + path)
    good = []
    for s in f:
        good.append(data_clean(s))
    jdump(good, '/zfsauton2/home/yifuc/11711-RAG/data/cmu/' + path)
    
        

