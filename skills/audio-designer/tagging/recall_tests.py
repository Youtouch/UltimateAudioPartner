"""Model-free catalogue and browser Y/N contracts."""
from pathlib import Path
import csv,json,sys,tempfile
from unittest.mock import patch
import recall_tags as r

def check():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        for name in ('Pack/UIClick_test.wav','Other/opaque.wav'):
            (root/name).parent.mkdir(exist_ok=True);(root/name).write_bytes(b'fixture')
        with (root/'AUDIO_LEDGER_files.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['relative_path','duration_seconds','pack']);w.writeheader()
            for name in ('Pack/UIClick_test.wav','Other/opaque.wav'):w.writerow(dict(relative_path=name,duration_seconds='.1',pack=name.split('/')[0]))
        with patch.dict(r.os.environ,{'HF_HUB_CACHE':str(root/'missing-models')}):
            out,_=r.build(root,progress=lambda x:None)
            found=r.query_catalog(root,'click');assert found['counts']['total']==1
            assert found['counts']['confirmed']==0 and not found['coverage']['calibrated']
            assert found['results'][0]['sources']==['name']
            assert len(r.category_tree(root))==753
            assert r.boundary_sample(root,'UIClick')[0]['path']=='Pack/UIClick_test.wav'
            import sound_browser
            lib=sound_browser.Library(root)
            answer=lib.set_category_answer({'path':'Pack/UIClick_test.wav','catid':'UIClick','answer':True})
            assert answer['saved']['answer'] is True
            assert lib.verdicts()['Pack/UIClick_test.wav']['category_answers']['UIClick']['answer'] is True
            try:r.build(root,out,progress=lambda x:None)
            except FileExistsError:pass
            else:raise AssertionError('Catalogue overwritten')
            (root/'Pack/UIClick_test.wav').write_bytes(b'changed fixture')
            assert r.query_catalog(root,'click')['results'][0]['stale']
    print('PASS category catalogue: no model cache, full candidates, honest evidence, Y/N persistence, immutable output, freshness')

if __name__=='__main__':check()
