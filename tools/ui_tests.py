"""Check waveform, comparison and keyboard HTML hooks without a browser install."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


def contract(html):
    for hook in ['id="inspector"', 'id="spectrogram"', 'id="comparebtn"',
                 'id="comparestrip"', 'id="matchloudness"', 'data-wave-path=',
                 '/api/spectrogram?p=', '/api/compare', 'createMediaElementSource',
                 'createGain', 'gain.setValueAtTime', 'requestAnimationFrame',
                 'id="shortcutguide"', 'function versionedUrl', 'async function refreshWaveform',
                 'function clearComparison']:
        assert hook in html, 'Missing browser hook: ' + hook
    script = re.search(r'<script>([\s\S]*?)</script>', html)[1]
    with tempfile.TemporaryDirectory(prefix='uap-ui-') as tmp:
        path = Path(tmp) / 'browser.js'
        path.write_text(script, encoding='utf-8')
        subprocess.run(['node', '--check', str(path)], check=True, capture_output=True)
    # Exercise the exact production functions and keyboard listener with tiny DOM stubs.
    helpers = script.split('// POLISH HELPERS START')[1].split('// POLISH HELPERS END')[0]
    keys = script.split("document.addEventListener('keydown', e => {")[1].split('\n});')[0]
    harness = r'''
const assert = require('node:assert/strict');
const esc = s => String(s).replaceAll('"', '&quot;');
let D={rows:[['a.wav','','',10,0,0,[],[],0]],waveforms:{'a.wav':{peaks:[0,.5,1],duration_s:10}}};
let current=0, playbackRequest=1, comparison={items:[{path:'a.wav',gain:.25}]}, context=null, gainNode=null;
const elements={'#matchloudness':{checked:true},'#avoid':{value:''},'#comparestrip':{hidden:false}};
const $=s=>elements[s];
const document={querySelector:()=>null,querySelectorAll:()=>[]};
const audio={currentTime:0,duration:10,dataset:{p:'a.wav'},paused:true,play(){this.paused=false},pause(){this.paused=true}};
const calls=[]; const play=(i,t)=>{calls.push(['play',i,t]);audio.currentTime=t};
const verdict=(i,v)=>calls.push(['verdict',i,v]); const step=d=>calls.push(['step',d]);
const match=(u,l)=>calls.push(['match',u]); const copy=i=>calls.push(['copy',i]);
const toast=t=>calls.push(['toast',t]); const picked=new Set();
const redraw=()=>{}; const updatePicked=()=>{};
const setPlaybackGain=()=>calls.push(['gain',gainForPlayback(audio.dataset.p)]);
let fetch=async()=>({ok:true,json:async()=>({peaks:[1],duration_s:2,source_stamp:'v3:96044:1791848720123456789'})});
'''
    assertions = r'''
(async()=>{
const svg=waveSvg('a.wav');
assert.match(svg,/data-wave-path="a.wav"/); assert.match(svg,/<path/);
assert.equal(gainForPlayback('a.wav'),.25);
elements['#matchloudness'].checked=false; assert.equal(gainForPlayback('a.wav'),1);
seekWave({dataset:{wavePath:'a.wav'},getBoundingClientRect:()=>({left:100,width:200})},150);
assert.equal(audio.currentTime,2.5);
const key=e=>{ KEY_BODY };
const event=k=>({key:k,target:{matches:()=>false},preventDefault(){}});
key(event('ArrowDown'));key(event('ArrowUp'));key(event('a'));key(event('R'));key(event('s'));
assert.deepEqual(calls.slice(-5).map(c=>c[0]),['step','step','verdict','verdict','match']);
assert.deepEqual(calls.slice(-3,-1),[['verdict',0,'like'],['verdict',0,'reject']]);
assert.match(calls.at(-1)[1],/^\/api\/like\?p=/);
evidence={0:{name:{rank:2}}};
assert.match(evidenceHtml(0),/name #2/);
assert.doesNotMatch(evidenceHtml(0),/audio|label/);
const before=calls.length;key({...event('a'),target:{matches:()=>true}}); assert.equal(calls.length,before);
elements['#matchloudness'].checked=true;
comparison={items:[{path:'a.wav',gain:.25,waveform:{source_stamp:'old'}}]};
assert.equal(gainForPlayback('a.wav'),.25);
assert.equal(await refreshWaveform(0,1),true);
assert.equal(comparison,null);assert.equal(gainForPlayback('a.wav'),1);
assert.ok(calls.some(c=>c[0]==='gain'&&c[1]===1));
assert.equal(D.waveforms['a.wav'].duration_s,2);
const firstUrl=versionedUrl('/audio?p=','a.wav');
assert.match(firstUrl,/1791848720123456789/);
assert.match(versionedUrl('/api/spectrogram?p=','a.wav'),/1791848720123456789/);
D.waveforms['a.wav'].source_stamp='v3:96044:1791848720123456790';
assert.notEqual(versionedUrl('/audio?p=','a.wav'),firstUrl);
let release;fetch=()=>new Promise(resolve=>release=resolve);
const stale=refreshWaveform(0,1);playbackRequest=2;
release({ok:true,json:async()=>({peaks:[0],duration_s:99,source_stamp:'obsolete'})});
assert.equal(await stale,false);assert.equal(D.waveforms['a.wav'].duration_s,2);
clearComparison();assert.equal(comparison,null);assert.equal(elements['#comparestrip'].hidden,true);
assert.deepEqual(calls.at(-1),['gain',1]);
console.log('PASS real UI helpers: envelope, seek, matched gain, keyboard and input guard');
console.log('PASS source freshness: exact source token, per-open refresh, request race guard, comparison reset');
})().catch(e=>{console.error(e);process.exit(1)});
'''.replace('KEY_BODY', keys)
    run = subprocess.run(['node', '-e', harness + helpers + assertions], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    print(run.stdout.strip())


def main():
    p = argparse.ArgumentParser(description=__doc__, epilog='Example: python tools/ui_tests.py')
    p.add_argument('--from', dest='source', type=Path, help='skill folder to check instead of the shipped skill')
    a = p.parse_args()
    source = a.source or Path(__file__).resolve().parents[1] / 'skills/audio-designer'
    html = (source / 'scripts/sound_browser.html').read_text(encoding='utf-8')
    contract(html)
    # Ensure each feature check detects a disabled implementation.
    for hook in ['id="spectrogram"', '/api/compare', 'createGain', 'id="shortcutguide"']:
        try:
            contract(html.replace(hook, 'disabled-feature'))
        except AssertionError:
            print('PASS UI mutation rejected: ' + hook)
        else:
            raise AssertionError('UI mutation survived: ' + hook)
    print('PASS browser polish hooks')


if __name__ == '__main__':
    main()
