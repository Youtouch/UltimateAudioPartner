"""Capture six real demo browser screens using an installed headless Microsoft Edge."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from urllib.request import urlopen

from demo import build_demo, skill_source


CAPTURE = r'''
import fs from 'node:fs';
import assert from 'node:assert/strict';
const [port,base,out] = process.argv.slice(1);
const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const target = targets.find(t => t.type === 'page');
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
let id=0;const pending=new Map(),errors=[];
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.text+' '+(m.params.exceptionDetails.exception?.description||''));if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});
const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
const waitFor=async expression=>{for(let k=0;k<150;k++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,100));}throw Error('Timed out: '+expression);};
await call('Runtime.enable');await call('Page.enable');
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
await call('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:'light'}]});
await call('Page.navigate',{url:base});
await waitFor("typeof D !== 'undefined' && D?.rows.length > 0 && document.querySelectorAll('.item').length > 0");
await evaluate("$('#cands').checked=true;$('#pack').value='Game SFX';apply()");
assert.equal(await evaluate("filtered.length"),4);
assert.equal(await evaluate("document.querySelectorAll('.item [data-wave-path]').length"),4);
assert.equal(await evaluate("$('#demobanner').hidden"),false);
await evaluate("$('#describe').value='short bright click';$('#go').click()");
await waitFor("categoryResult !== null && document.querySelectorAll('.item').length > 0 && $('#categorysummary').textContent.includes('2')");
assert.deepEqual(errors,[]);
const evidence=[];
const shot=async(name,check)=>{await waitFor(check);await waitFor("!$('#toast').classList.contains('show')");await new Promise(r=>setTimeout(r,250));const p=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(out+'/'+name+'.png',Buffer.from(p.data,'base64'));evidence.push({file:name+'.png',url:base,assertion:check});console.log('PASS real UI screenshot '+name);};
await evaluate("$('#pack').value='';apply();document.querySelectorAll('#categorytree details').forEach(d=>{if(d.textContent.includes('USER INTERFACE')){d.open=true;d.scrollIntoView({block:'center'})}})");
await shot('category-recall',"categoryResult.counts.total === 2 && categoryResult.results.every(r=>!r.calibrated && !r.stale) && categoryResult.coverage.stale === false && !$('#categorycoverage').textContent.includes('out of date') && document.querySelectorAll('.item').length === 2");
await evaluate("$('#clearscore').click();$('#describe').value='';$('#pack').value='Game SFX';apply()");
await shot('library',"document.querySelectorAll('.item .chip').length > 4");
await evaluate("play(D.rows.findIndex(r=>r[0].includes('laser_sweep')))");
await waitFor("$('#spectrogram').complete && $('#spectrogram').naturalWidth > 0 && $('#detailnumbers').children.length >= 4");
await evaluate("audio.pause();audio.currentTime=.25;updatePlayheads()");
assert.equal(await evaluate("$('#detailwave .wavehead').hidden"),false);
// Exercise the actual waveform hit target rather than setting a fake visual state.
await evaluate("{const w=$('#detailwave .wave'),b=w.getBoundingClientRect();w.dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:b.left+b.width/2}));}");
assert.ok(await evaluate("Math.abs(audio.currentTime-.325)<.02"));
await shot('sound-detail',"!$('#inspector').hidden && $('#spectrogram').naturalWidth > 0");
await evaluate("setExplain(true)");
await shot('explain',"document.body.classList.contains('explain') && $('#explainbox').textContent.includes('attack')");
await evaluate("setExplain(false);$('#closedetail').click();filtered.slice(0,3).forEach(i=>document.querySelector(`.item[data-i=\"${i}\"] .sel`).click());$('#comparebtn').click()");
await shot('compare',"!$('#comparestrip').hidden && document.querySelectorAll('.comparecard').length === 3");
await evaluate("document.querySelector('[data-compare-play]').click()");
await waitFor("gainNode !== null && audio.dataset.p === comparison.items[0].path && Math.abs(gainNode.gain.value-comparison.items[0].gain)<.00001 && !audio.paused");
assert.ok(await evaluate("Math.abs(gainNode.gain.value-comparison.items[0].gain)<.00001"));
await evaluate("$('#matchloudness').checked=false;$('#matchloudness').dispatchEvent(new Event('change'))");
await waitFor("gainNode.gain.value === 1");
assert.equal(await evaluate("gainNode.gain.value"),1);
await evaluate("audio.pause();$('#matchloudness').checked=true;$('#matchloudness').dispatchEvent(new Event('change'));$('#closecompare').click();$('#closedetail').click();$('#exportbtn').click();$('#exportout').value='C:/MyGame/Assets/Audio/SFX'");
await shot('export',"$('#exportdlg').open && $('#exportcount').textContent.includes('3')");
assert.deepEqual(errors,[]);
fs.writeFileSync(out+'/capture-evidence.json',JSON.stringify({engine:'Microsoft Edge headless / DevTools Protocol',synthetic:true,screenshots:evidence,checks:['real UI loaded','four cached row envelopes','complete category count with two generated clicks','current recall source stamps and no stale evidence','unsupported toy precision labelled uncalibrated','spectrogram decoded','click seeks to half duration','comparison applies measured gain','toggle returns to unity','no JavaScript exceptions']},null,2));
ws.close();
'''


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def compact_png(path):
    """Recompress PNG scanlines losslessly with the standard library."""
    body = path.read_bytes()
    chunks, offset = [], 8
    while offset < len(body):
        length = struct.unpack('>I', body[offset:offset + 4])[0]
        kind = body[offset + 4:offset + 8]
        chunks.append((kind, body[offset + 8:offset + 8 + length]))
        offset += length + 12
    compressed = zlib.compress(zlib.decompress(b''.join(data for kind, data in chunks if kind == b'IDAT')), 9)
    parts, written = [body[:8]], False
    for kind, data in chunks:
        if kind == b'IDAT':
            if written:
                continue
            data, written = compressed, True
        parts.append(struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data)))
    result = b''.join(parts)
    if len(result) < len(body):
        path.write_bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Example: python tools/capture_demo.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'docs/img', help='screenshot folder')
    parser.add_argument('--edge', type=Path, help='installed Microsoft Edge executable')
    parser.add_argument('--library', type=Path, help='existing synthetic demo; otherwise build a new temporary demo')
    args = parser.parse_args()
    edge = args.edge or shutil.which('msedge')
    if not edge:
        for folder in ('PROGRAMFILES(X86)', 'PROGRAMFILES'):
            path = Path(os.environ.get(folder, '')) / 'Microsoft/Edge/Application/msedge.exe'
            if path.is_file():
                edge = path
                break
    if not edge or not shutil.which('node'):
        parser.error('Installed Microsoft Edge and Node.js are required; nothing is downloaded.')
    source = skill_source()
    sys.path.insert(0, str(source / 'scripts'))
    import demo_metadata
    import sound_browser
    root = args.library or build_demo(source=source)
    if not demo_metadata.read_demo(root):
        parser.error('Screenshots require a synthetic UAP demo library.')
    args.out.mkdir(parents=True, exist_ok=True)
    server = sound_browser.Server(('127.0.0.1', 0), sound_browser.handler(sound_browser.Library(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = 'http://127.0.0.1:' + str(server.server_address[1])
    port = free_port()
    try:
        with tempfile.TemporaryDirectory(prefix='uap-edge-') as profile:
            command = [str(edge), '--headless=new', '--no-first-run', '--no-default-browser-check',
                       '--disable-background-networking', '--disable-component-update', '--disable-sync',
                       '--disable-extensions', '--autoplay-policy=no-user-gesture-required',
                       '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=' + str(port),
                       '--user-data-dir=' + profile, base]
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                for _ in range(100):
                    try:
                        with urlopen(f'http://127.0.0.1:{port}/json/version', timeout=1):
                            break
                    except OSError:
                        time.sleep(.1)
                else:
                    raise RuntimeError('Headless Edge did not start')
                subprocess.run(['node', '--input-type=module', '-e', CAPTURE, str(port), base, str(args.out.resolve())],
                               check=True, timeout=120)
            finally:
                process.terminate()
                process.wait(timeout=15)
    finally:
        server.shutdown()
        server.server_close()
    for path in sorted(args.out.glob('*.png')):
        compact_png(path)
        if path.stat().st_size >= 300_000:
            raise RuntimeError(f'Screenshot exceeds 300 KB: {path}')
        print(f'PASS {path.name}: {path.stat().st_size:,} bytes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
