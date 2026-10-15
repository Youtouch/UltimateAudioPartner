"""Offline regressions for official identities, honest mappings and lexical recall.

The mutation checks execute the same behavioral contract with deliberately
broken public functions. A passing check must reject every mutation.

    python tagging/ucs_tests.py
"""
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from unittest.mock import patch
sys.dont_write_bytecode = True
import ucs


def contract(api):
    taxonomy=api.load_taxonomy()
    assert len(taxonomy)==753 and len({r['category'] for r in taxonomy})==82
    ids={r['catid'] for r in taxonomy}
    assert 'WOODHndl' in ids and 'WOODHandl' not in ids,'official corrected CatID'
    assert {'UIClick','MECHClik','CMPTKey','MECHSwtch'}<=set(api.resolve_query('short bright click')['catids'])
    assert api.resolve_query('short bright click')['remaining_query']=='short bright'
    assert set(api.resolve_query('UI')['catids'])=={r['catid'] for r in taxonomy if r['category']=='USER INTERFACE'}
    assert api.resolve_query('UIClick')['catids']==['UIClick']
    assert set(api.resolve_query('coin/pickup')['catids'])=={'OBJCoin','UIMisc'}
    assert set(api.resolve_query('button/UI select')['catids'])=={'UIClick','UIBeep','UIMisc'}
    assert api.resolve_query('button/UI select')['remaining_query']==''
    assert set(api.resolve_query('FOOTSTEPS')['catids'])=={'FEETHmn','FEETHors','FEETAnml','FEETCrea','FEETMisc'}
    assert api.resolve_query('unclassified texture with nothing identifiable')['catids']==[]
    assert not api.weak_matches('Click Pack/Unknown/quux_03.wav'), 'parent and pack names hidden'
    assert not api.weak_matches('Pack/clickbait.wav'), 'whole words, no substring label leakage'
    assert not api.weak_matches('Pack/no_click.wav'), 'negated names are not positive truth'
    assert 'UIClick' in api.weak_matches('Pack/UI_Click_01.wav')
    assert 'UIClick' in api.weak_matches('Pack/btn_click_02.wav')
    assert api.weak_matches(r'Pack\MouseClick_03.wav')=={'CMPTKey'}, 'camelCase has specific identity'
    assert 'CMPTKey' in api.weak_matches('Pack/Keyboard_04.wav')
    assert 'MECHSwtch' in api.weak_matches('Pack/Switch_05.wav')
    assert 'FEETHmn' in api.weak_matches('Pack/footsteps_on_gravel.wav')
    assert 'FEETHors' in api.weak_matches('Pack/HorseFootsteps.wav')
    # Official IDs give filename evidence for every subcategory, including ones
    # with no good natural-language examples in a particular library.
    for row in taxonomy:
        assert row['catid'] in api.weak_matches('UnknownPack/'+row['catid']+'_03.wav'),row['catid']
    return 24+len(taxonomy)


def alias_contract(resolver):
    aliases={'florble':['UIClick'],'dense florble':['MECHClik']}
    assert resolver('short bright florble',aliases=aliases)['catids']==['UIClick']
    assert resolver('short bright florble',aliases=aliases)['remaining_query']=='short bright'
    assert resolver('dense florble',aliases=aliases)['catids']==['MECHClik'], 'longest personal phrase leads'
    assert resolver('florblebait',aliases=aliases)['catids']==[], 'personal words retain whole-term boundaries'
    assert resolver('no florble',aliases=aliases)['catids']==[], 'negated personal labels are not positive categories'
    assert resolver('UIClick',aliases={'UIClick':['MECHClik']})['catids']==['UIClick'], 'explicit official IDs remain stable'


def check_aliases():
    alias_contract(ucs.resolve_query)
    try:alias_contract(lambda query,aliases=None:ucs.resolve_query(query))
    except AssertionError:pass
    else:raise AssertionError('Ignoring personal aliases escaped mutation contract')
    with tempfile.TemporaryDirectory(prefix='uap-alias-test-') as temporary:
        root=Path(temporary)
        assert ucs.read_aliases(root)=={} and not list(root.iterdir()), 'optional alias read has no write side effects'
        audio=root/'original.wav';audio.write_bytes(b'original-audio-fixture')
        ucs.add_alias(root,' Blip! ',['uiclick'])
        ucs.add_alias(root,'whorl',['WHSH'])
        ucs.add_alias(root,'blip',['MECHClik'])
        aliases=ucs.read_aliases(root)
        assert aliases=={'blip':['MECHClik','UIClick'],'whorl':['WHSH']}, 'updates preserve other aliases and merge same word'
        sidecar=root/ucs.ALIASES_FILE;before=sidecar.read_bytes()
        for word,ids in [('123', ['UIClick']),('blip',['invented-id']),('blip',[])]:
            try:ucs.add_alias(root,word,ids)
            except ValueError:pass
            else:raise AssertionError('Invalid personal alias accepted')
            assert sidecar.read_bytes()==before, 'invalid alias cannot change existing sidecar'
        with patch.object(ucs.os,'replace',side_effect=OSError('simulated publication failure')):
            try:ucs.add_alias(root,'newword',['UIAlert'])
            except OSError:pass
            else:raise AssertionError('Atomic-write failure vanished')
        assert sidecar.read_bytes()==before and not list(root.glob('.uap-category-alias-*.tmp'))
        result=subprocess.run([sys.executable,str(Path(ucs.__file__)),'alias','--root',str(root),
          '--word','chirplet','--catid','UIClick','--catid','MECHClik'],capture_output=True,text=True)
        assert result.returncode==0, result.stderr
        assert ucs.read_aliases(root)['chirplet']==['MECHClik','UIClick']
        assert audio.read_bytes()==b'original-audio-fixture'
    return 15


def check():
    count=contract(ucs)
    alias_checks=check_aliases()
    data=ucs.load_bundle()
    assert 'audioset_ontology' not in data and 'ced_display_labels' not in data
    ontology=set(ucs.cached_class_names('MIT/ast-finetuned-audioset-10-10-0.4593'))
    assert data['official_fields']==['catid','category','subcategory','catshort','explanation','synonyms']
    omissions=data['publication_omissions']
    assert omissions['count']==1 and omissions['field']=='synonyms' and omissions['catids']==['OBJBook']
    assert omissions['reason'] and data['source_sha256'], 'official-source hash and omission provenance retained'
    assert len(data['alias_mappings'])==115
    assert sum(bool(x['catids']) for x in data['alias_mappings'].values())==108
    for row in data['categories']:
        assert 3<=len(row['prompts'])<=5 and len(set(row['prompts']))==len(row['prompts'])
        assert not ontology or set(row['audioset_labels'])<=ontology
        assert row['synonyms'] and row['explanation'],row['catid']
        if not row['audioset_labels']:
            assert row['audioset_mapping_status']=='unmapped_no_confident_ontology_match'
    assert ucs.label_variants('Walk, footsteps')==['Walk, footsteps','Walk']
    assert ucs.label_variants('Gunshot, gunfire')==['Gunshot, gunfire','Gunshot']
    with patch.object(ucs,'cached_class_names',side_effect=lambda model:['Walk','Gunshot'] if model=='mispeech/ced-base' else []):
        assert ucs.label_variants('Walk, footsteps')==['Walk, footsteps','Walk']
        assert ucs.label_variants('Uncached, other')==['Uncached, other']
    mutations={
       'folder-label leakage': {'weak_matches':lambda path,taxonomy=None:ucs.weak_matches(str(path).replace('/','_').replace('\\','_'),taxonomy)},
       'lost camel-case specificity': {'weak_matches':lambda path,taxonomy=None:ucs.weak_matches(str(path).lower(),taxonomy)},
       'truncated category family': {'resolve_query':lambda query:dict(ucs.resolve_query(query),catids=ucs.resolve_query(query)['catids'][:1])},
       'dropped refinements': {'resolve_query':lambda query:dict(ucs.resolve_query(query),remaining_query='')},
       'lost composite alias': {'resolve_query':lambda query:dict(ucs.resolve_query(query),catids=['OBJCoin']) if query=='coin/pickup' else ucs.resolve_query(query)},
       'substring truth': {'weak_matches':lambda path,taxonomy=None:({'UIClick'} if 'click' in str(path).lower() else ucs.weak_matches(path,taxonomy))},
    }
    killed=[]
    for name,overrides in mutations.items():
        class Broken:
            load_taxonomy=staticmethod(ucs.load_taxonomy)
            resolve_query=staticmethod(overrides.get('resolve_query',ucs.resolve_query))
            weak_matches=staticmethod(overrides.get('weak_matches',ucs.weak_matches))
        try:contract(Broken)
        except AssertionError:killed.append(name)
        else:raise AssertionError('Survived broken-version mutation: '+name)
    print(json.dumps({'status':'PASS','behavior_assertions':count,'ucs_subcategories':753,
      'ucs_categories':82,'audioset_mapped':sum(bool(r['audioset_labels']) for r in data['categories']),
      'audioset_unmapped':sum(not r['audioset_labels'] for r in data['categories']),
      'legacy_aliases':115,'legacy_aliases_mapped':108,'personal_alias_checks':alias_checks,
      'mutations_rejected':killed+['ignored personal aliases']},indent=2))
    return {'mutations_rejected':killed}


if __name__=='__main__':check()
