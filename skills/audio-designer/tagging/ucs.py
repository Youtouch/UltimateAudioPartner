"""Official UCS 8.2.1 identities with explicitly authored recognition mappings.

`load_taxonomy` is offline and reads the public-domain list bundled in
ucs_mapping.json. Names, explanations and synonyms are UCS fields; prompts,
AudioSet correspondences and aliases are UAP proposals. Empty AudioSet lists
are intentionally unmapped. No filenames or library data are bundled here.

`weak_matches` uses only the basename. It is noisy weak evidence, never a
listening verdict. Broad/ambiguous UCS synonyms are excluded unless qualified.
`resolve_query` also expands source families across related UCS CatIDs. UCS
itself has no footstep surface children: material words remain refinements.

    python tagging/ucs.py check
    python tagging/ucs.py resolve "short bright click"
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MAPPING = HERE / 'ucs_mapping.json'
ALIASES_FILE = 'AUDIO_CATEGORY_ALIASES.json'
NEGATIONS = {'no', 'not', 'without', 'except', 'excluding'}
DESCRIPTORS = {'short', 'long', 'bright', 'dark', 'soft', 'loud', 'quiet', 'punchy',
               'sharp', 'dull', 'single', 'repeated', 'fast', 'slow', 'tonal', 'noisy',
               'loop', 'loopable', 'dry', 'wet', 'crisp', 'gentle', 'deep', 'texture'}
STOP = {'a', 'an', 'the', 'of', 'and', 'sound', 'sounds', 'effect', 'effects', 'unknown', 'unclassified',
        'misc', 'general', 'source', 'real', 'handle', 'movement', 'designed'}


def tokens(text):
    """Whole words, camel/acronym boundaries and digits; never substring matches."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(text))
    text = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', text)
    return tuple(x for x in re.findall(r'[^\W\d_]+', text.casefold(), flags=re.UNICODE) if x)


def _singular(word):
    # Conservative inflections only; 'glass', 'bass', 'gas' must survive intact.
    fixed = {'footsteps':'footstep', 'feet':'footstep', 'clicking':'click',
             'clicks':'click', 'beeps':'beep', 'coins':'coin', 'gunshots':'gunshot',
             'whooshes':'whoosh', 'swooshes':'swoosh', 'swishes':'swish',
             'notifications':'notification', 'buttons':'button', 'impacts':'impact',
             'explosions':'explosion', 'lasers':'laser', 'steps':'step'}
    return fixed.get(word, word)


def _normal(text):
    return tuple(_singular(x) for x in tokens(text))


@lru_cache(maxsize=1)
def load_bundle():
    """Return provenance plus mapping, validating identities and ontology references."""
    data = json.loads(MAPPING.read_text(encoding='utf-8'))
    entries = data['categories']
    ids = [r['catid'] for r in entries]
    if len(entries) != 753 or len(set(ids)) != len(ids):
        raise ValueError('UCS 8.2.1 must contain 753 distinct CatIDs')
    ontology = set(cached_class_names('MIT/ast-finetuned-audioset-10-10-0.4593'))
    for row in entries:
        if not 3 <= len(row['prompts']) <= 5:
            raise ValueError('Each UCS subcategory needs 3-5 authored prompts')
        if ontology and not set(row['audioset_labels']) <= ontology:
            raise ValueError('Mapping references a label outside the AudioSet ontology')
        if not row['audioset_labels'] and row['audioset_mapping_status'] != 'unmapped_no_confident_ontology_match':
            raise ValueError('Empty AudioSet mapping must be explicitly unmapped')
    return data


def load_taxonomy():
    """Return the 753 UCS subcategory dictionaries (treat their contents as read-only)."""
    return load_bundle()['categories']


def _validated_aliases(aliases, taxonomy=None):
    if not isinstance(aliases, dict):
        raise ValueError('Category aliases must be an object mapping words to CatID lists')
    taxonomy=load_taxonomy() if taxonomy is None else taxonomy
    official={r['catid'].casefold():r['catid'] for r in taxonomy}
    normalized={}
    for word,catids in aliases.items():
        if not isinstance(word,str) or not _normal(word):
            raise ValueError('Category alias words must contain at least one letter')
        if not isinstance(catids,list) or not catids:
            raise ValueError('Each category alias requires a nonempty CatID list')
        ids=[]
        for cid in catids:
            if not isinstance(cid,str) or cid.casefold() not in official:
                raise ValueError('Category alias references an unknown UCS CatID')
            ids.append(official[cid.casefold()])
        key=' '.join(_normal(word))
        normalized[key]=sorted(set(normalized.get(key,[])+ids))
    return normalized


def read_aliases(root):
    """Read optional library words without creating metadata or changing files."""
    path=Path(root)/ALIASES_FILE
    if not path.exists():return {}
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data,dict) or data.get('schema')!=1 or 'aliases' not in data:
        raise ValueError('Unsupported category-alias sidecar schema')
    return _validated_aliases(data['aliases'])


def add_alias(root,word,catids):
    """Merge a personal word using an atomic replacement of its own sidecar."""
    root=Path(root)
    if not root.is_dir():raise ValueError('Category alias root must be an existing library folder')
    addition=_validated_aliases({word:catids})
    aliases=read_aliases(root)
    for key,ids in addition.items():aliases[key]=sorted(set(aliases.get(key,[])+ids))
    path=root/ALIASES_FILE
    descriptor,temporary=tempfile.mkstemp(prefix='.uap-category-alias-',suffix='.tmp',dir=root)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8',newline='\n') as stream:
            json.dump({'schema':1,'aliases':aliases},stream,indent=2,ensure_ascii=False,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return aliases


def cached_class_names(model):
    """Read id2label locally without importing/loading a model or accessing the network."""
    cache = Path(os.environ.get('HF_HUB_CACHE') or
                 Path(os.environ.get('HF_HOME', Path.home()/'.cache/huggingface'))/'hub')
    snapshots = cache/('models--'+model.replace('/', '--'))/'snapshots'
    for config in sorted(snapshots.glob('*/config.json'), key=lambda p:p.stat().st_mtime_ns, reverse=True):
        try:
            labels=json.loads(config.read_text(encoding='utf-8'))['id2label']
            return [labels[str(i)] for i in range(len(labels))]
        except (OSError,ValueError,KeyError,TypeError):
            continue
    return []


def label_variants(label):
    """Match display names by output ordinal only when both local configs exist."""
    canonical=cached_class_names('MIT/ast-finetuned-audioset-10-10-0.4593')
    display=cached_class_names('mispeech/ced-base')
    try:
        i=canonical.index(label)
    except ValueError:
        # A standard CED-only install has no AST config. Its shortened display
        # name may match an authored reference prefix, only if that name exists
        # in the cached CED config. Without cache, retain the authored reference.
        prefix=label.split(',',1)[0].strip()
        return list(dict.fromkeys([label,prefix])) if prefix in display else [label]
    return list(dict.fromkeys([label,display[i]])) if i<len(display) else [label]


def _family(taxonomy):
    bycat = defaultdict(set)
    for row in taxonomy:
        bycat[row['category']].add(row['catid'])
    allids = {r['catid'] for r in taxonomy}
    values = {
        'click': {'UIClick', 'CMPTKey', 'MECHClik', 'MECHSwtch'},
        'footstep': bycat['FOOTSTEPS'],
        'whoosh': bycat['SWOOSHES'] | {'DSGNWhsh', 'FIREWhsh'},
        'impact': {r['catid'] for r in taxonomy if r['subcategory'] == 'IMPACT'},
        'coin': {'OBJCoin'}, 'pickup': {'UIMisc'},
        'coin pickup': {'OBJCoin','UIMisc'},
        'button ui select': {'UIClick','UIBeep','UIMisc'},
        'gunshot': bycat['GUNS'] - {'GUNHndl', 'GUNMech'},
        'explosion': bycat['EXPLOSIONS'], 'laser': bycat['LASERS'],
        'notification': {'UIAlert'}, 'button': {'UIClick', 'UIBeep', 'UIMisc'},
        'ui': bycat['USER INTERFACE'],
    }
    synonyms = {'swoosh':'whoosh', 'swish':'whoosh', 'footsteps':'footstep',
                'step':'footstep', 'select':'button', 'ui select':'button',
                'button select':'button', 'ui button':'button', 'btn':'button',
                'coin collect':'coin pickup', 'collect':'pickup',
                'power up':'pickup', 'alert':'notification', 'notify':'notification',
                'interface':'ui', 'user interface':'ui', 'gun fire':'gunshot',
                'gunfire':'gunshot', 'foot fall':'footstep'}
    out = {_normal(k): v & allids for k, v in values.items()}
    for term, canonical in synonyms.items():out[_normal(term)] = values[canonical] & allids
    return out


# Compiling is cheap but processing a whole library should compile only once.
_COMPILED = {}
def _compile(taxonomy):
    key = id(taxonomy)
    if key in _COMPILED and _COMPILED[key][0] is taxonomy:
        return _COMPILED[key][1]
    index = defaultdict(set)
    official = defaultdict(set)
    bycat = defaultdict(list)
    byid = {r['catid']:r for r in taxonomy}
    aliases = defaultdict(set)
    def add(term, cid):
        pattern = _normal(term)
        if pattern:index[pattern].add(cid)
    for row in taxonomy:
        cid,cat,sub = row['catid'],row['category'],row['subcategory']
        bycat[cat].append(row)
        add(cid,cid)
        add(cat+' '+sub,cid)
        if '&' in sub:
            for part in sub.split('&'):add(cat+' '+part.strip(),cid)
        for term in row['synonyms']:official[_normal(term)].add(cid)
        for term in row['aliases']:aliases[term].add(cid)
    # Unqualified subcategory terms must be distinctive. 'HUMAN' is not footsteps.
    subcategories = defaultdict(set)
    for row in taxonomy:subcategories[_normal(row['subcategory'])].add(row['catid'])
    for pattern,catids in subcategories.items():
        if len(catids)==1 and not set(pattern)&(STOP|DESCRIPTORS) and pattern not in {('human',),('animal',),('bird',),('insect',)}:
            index[pattern].update(catids)
    # Expanded UCS synonyms contain general associations (e.g. air, movement,
    # bright). Only globally unambiguous, concrete terms are weak truth.
    for pattern,catids in official.items():
        if len(catids)==1 and pattern and not set(pattern)&(STOP|DESCRIPTORS) and not any(len(t)<3 for t in pattern):
            index[pattern].update(catids)
    for term,catids in aliases.items():
        if len(catids)<=3:
            for cid in catids:
                add(term,cid)
                row=byid[cid]
                # Legacy synonyms only if their tag's target is sufficiently narrow.
                for name in row.get('alias_name_terms',[]):
                    # Avoid broad legacy terms inherited by another alias on row.
                    if _normal(name) not in {('music',),('sound',),('effect',)}:add(name,cid)
    # Parent-only names give a conservative miscellaneous/general child. They
    # cannot establish every child (e.g. "wood" is not every wood action).
    for cat,children in bycat.items():
        default=next((r for r in children if r['subcategory']=='GENERAL'),None)
        default=default or next((r for r in children if r['subcategory']=='MISC'),None)
        if default:add(cat,default['catid'])
    # Reviewed common filename spellings, with precise UCS identities.
    fixed = {
      'click':['MECHClik'], 'ui click':['UIClick'], 'ui':['UIMisc'],
      'button click':['UIClick'], 'btn click':['UIClick'], 'button':['UIClick'],
      'ui select':['UIClick'], 'ui confirm':['UIBeep'], 'ui cancel':['UIMisc'],
      'ui hover':['UIClick'], 'ui back':['UIMvmt'],
      'mouse click':['CMPTKey'], 'keyboard':['CMPTKey'],
      'switch':['MECHSwtch'], 'footstep':['FEETHmn'], 'foot steps':['FEETHmn'],
      'footsteps on gravel':['FEETHmn'], 'footsteps on grass':['FEETHmn'],
      'whoosh':['WHSH'], 'swoosh':['WHSH'], 'swish':['SWSH'],
      'impact':['DSGNImpt'], 'coin':['OBJCoin'], 'pickup':['UIMisc'],
      'pick up':['UIMisc'], 'power up':['UIMisc'], 'notification':['UIAlert'],
      'gunshot':['GUNMisc'], 'gun fire':['GUNMisc'], 'gunfire':['GUNMisc'],
      'explosion':['EXPLMisc'], 'laser':['LASRMisc'], 'zap':['ELECZap','LASRGun'],
      'footstep human':['FEETHmn'], 'footstep horse':['FEETHors'],
      'footstep animal':['FEETAnml'], 'footstep creature':['FEETCrea'],
      'horse footstep':['FEETHors'], 'animal footstep':['FEETAnml'],
      'creature footstep':['FEETCrea'], 'dog footstep':['FEETAnml'],
    }
    for term,catids in fixed.items():index[_normal(term)] = set(catids)&set(byid)
    result={'index':dict(index),'max_words':max(map(len,index)),
            'bycat':{k:{r['catid'] for r in v} for k,v in bycat.items()},
            'byid':byid,'families':_family(taxonomy)}
    _COMPILED[key]=(taxonomy,result)
    return result


def _matches(words,index,max_words,negations=True):
    hits=[]
    for start in range(len(words)):
        if negations and set(words[max(0,start-2):start])&NEGATIONS:continue
        for width in range(min(max_words,len(words)-start),0,-1):
            part=words[start:start+width]
            if part in index:hits.append((start,start+width,part,set(index[part])))
    return hits


def weak_matches(path,taxonomy=None):
    """Filename-only CatID candidates; parent directories never supply truth.

    Longest overlapping term wins: MouseClick is keyboard/mouse, while plain
    click is mechanical. Negated names (no_click) are excluded. Explicit UCS
    prefixes are authoritative filename evidence, still not listening truth.
    """
    taxonomy=load_taxonomy() if taxonomy is None else taxonomy
    compiled=_compile(taxonomy)
    basename=str(path).replace('\\','/').rsplit('/',1)[-1].rsplit('.',1)[0]
    # Recognize the case-insensitive CatID as a full filename segment.
    segments=re.split(r'[_\s.\-]+',basename.casefold())
    explicit={cid for cid in compiled['byid'] if cid.casefold() in segments}
    if explicit:return explicit
    words=_normal(basename)
    hits=_matches(words,compiled['index'],compiled['max_words'])
    # Generic terms wholly contained in a more specific match add no labels.
    maximal=[h for h in hits if not any(o[0]<=h[0] and o[1]>=h[1] and o[1]-o[0]>h[1]-h[0] for o in hits)]
    return set().union(*(h[3] for h in maximal)) if maximal else set()


def resolve_query(query,taxonomy=None,aliases=None):
    """Resolve a UCS CatID, parent name, family, alias or distinctive synonym.

    The remaining words are refinements inside the complete category set.
    Families are UAP cross-category expansions, not invented UCS children.
    """
    taxonomy=load_taxonomy() if taxonomy is None else taxonomy
    compiled=_compile(taxonomy)
    normalized=_normal(query)
    custom={_normal(word):set(ids) for word,ids in _validated_aliases(aliases if aliases is not None else {},taxonomy).items()}
    customhits=_matches(normalized,custom,max(map(len,custom),default=1))
    empty={'catids':[],'label':'','remaining_query':str(query).strip(),'matched_terms':[]}
    if not normalized:return empty
    for cid,row in compiled['byid'].items():
        if str(query).strip().casefold()==cid.casefold():
            return {'catids':[cid],'label':row['category']+' › '+row['subcategory'],
                    'remaining_query':'','matched_terms':[cid]}
    for cat,catids in compiled['bycat'].items():
        if normalized==_normal(cat) and not customhits:
            return {'catids':sorted(catids),'label':cat,'remaining_query':'','matched_terms':[cat]}
    # Family terms lead unless a qualified, narrower source phrase is present.
    familyhits=_matches(normalized,compiled['families'],4)
    hits=_matches(normalized,compiled['index'],compiled['max_words'])
    best_family=max(familyhits,key=lambda h:h[1]-h[0],default=None)
    best=max(hits,key=lambda h:h[1]-h[0],default=None)
    use=best_family if best_family and (not best or best[1]-best[0]<=best_family[1]-best_family[0]) else best
    best_custom=max(customhits,key=lambda h:h[1]-h[0],default=None)
    if best_custom and (not use or best_custom[1]-best_custom[0]>=use[1]-use[0]):use=best_custom
    if not use:return empty
    start,end,term,catids=use
    # A category qualifier, such as "UI click", can narrow a family.
    consumed=set(range(start,end))
    for cat,children in compiled['bycat'].items():
        catwords=_normal(cat)
        for i in range(len(normalized)-len(catwords)+1):
            if normalized[i:i+len(catwords)]==catwords and catids&children:
                catids=catids&children;consumed.update(range(i,i+len(catwords)))
    rows=[compiled['byid'][cid] for cid in sorted(catids)]
    if len(rows)==1:label=rows[0]['category']+' › '+rows[0]['subcategory']
    elif rows and len({r['category'] for r in rows})==1:label=rows[0]['category']+' ('+str(len(rows))+' subcategories)'
    else:label=' '.join(term)+' ('+str(len(rows))+' related UCS subcategories)'
    return {'catids':sorted(catids),'label':label,
            'remaining_query':' '.join(w for i,w in enumerate(tokens(query)) if i not in consumed),
            'matched_terms':[' '.join(term)]}


def check():
    from ucs_tests import check as run_checks
    return run_checks()


def main():
    parser=argparse.ArgumentParser(description='Resolve sound categories through the official UCS taxonomy.',
      epilog='Example: python tagging/ucs.py resolve "short bright click"')
    sub=parser.add_subparsers(dest='cmd',required=True,title='commands',help='available taxonomy commands')
    sub.add_parser('check',help='Run offline taxonomy and filename-truth regression checks.',
      description='Run offline taxonomy and filename-truth regression checks.',epilog='Example: python tagging/ucs.py check')
    resolve=sub.add_parser('resolve',help='Resolve a category word and retain its refinements.',
      description='Resolve a category word and retain its refinements.',epilog='Example: python tagging/ucs.py resolve "short click"')
    resolve.add_argument('query',help='category, UCS CatID or a descriptive sound query')
    alias=sub.add_parser('alias',help='Save a personal category word in a library.',
      description='Save a personal category word while preserving existing aliases.',
      epilog='Example: python tagging/ucs.py alias --root MY_LIBRARY --word blip --catid UIClick')
    alias.add_argument('--root',type=Path,required=True,help='existing library folder for the personal alias sidecar')
    alias.add_argument('--word',required=True,help='personal category word or phrase, matched as whole words')
    alias.add_argument('--catid',action='append',required=True,help='official UCS CatID; repeat to include related subcategories')
    args=parser.parse_args()
    if args.cmd=='check':return check()
    if args.cmd=='alias':
        try:result=add_alias(args.root,args.word,args.catid)
        except (ValueError,OSError) as exc:parser.error(str(exc))
        print(json.dumps({'aliases':result},ensure_ascii=False,indent=2))
        return
    print(json.dumps(resolve_query(args.query),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
