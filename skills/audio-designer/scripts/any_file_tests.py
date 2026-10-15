"""Generated audio edge cases through the real tag, search and browser paths."""
from pathlib import Path
import csv,json,sys,tempfile,struct,os
import numpy as np
import soundfile as sf
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import audio_tag,audio_search,sound_browser

def fixtures(root):
 def tone(rate=16000,channels=1,seconds=.2):
  y=(.2*np.sin(2*np.pi*440*np.arange(int(rate*seconds))/rate)).astype('float32')
  return np.tile(y[:,None],(1,channels)) if channels>1 else y
 cases={'ten_ms':(16000,'PCM_16',1,.01),'silence':(16000,'PCM_16',1,.5),'eight_bit':(16000,'PCM_U8',1,.2),'float32':(16000,'FLOAT',1,.2),'eight_k':(8000,'PCM_16',1,.2),'192_k':(192000,'PCM_16',1,.2),'mono':(16000,'PCM_16',1,.2),'surround_5_1':(48000,'PCM_16',6,.2),'non_ascii_é_音':(16000,'PCM_16',1,.2)}
 for name,(rate,sub,ch,sec) in cases.items():sf.write(root/(name+'.wav'),np.zeros(int(rate*sec)) if name=='silence' else tone(rate,ch,sec),rate,subtype=sub)
 opposed=tone(48000,1,.2);sf.write(root/'opposed_5_1.wav',np.column_stack([opposed,-opposed,opposed,-opposed,opposed,-opposed]),48000,subtype='FLOAT')
 long=root/('long_'+('a'*110))/('nested_'+('b'*110))/('sound_'+('c'*70)+'.wav');long.parent.mkdir(parents=True);sf.write(long,tone(),16000)
 # Sparse valid two-hour PCM container; no giant in-memory array.
 n=8000*7200*2;p=root/'two_hours.wav'
 with p.open('wb') as f:
  f.write(struct.pack('<4sI4s4sIHHIIHH4sI',b'RIFF',n+36,b'WAVE',b'fmt ',16,1,1,8000,16000,2,16,b'data',n));f.seek(n+43);f.write(b'\0')
 late=np.zeros(65*16000,'float32');late[-16000:]=tone(seconds=1);sf.write(root/'late_event.wav',late,16000)
 (root/'corrupt.wav').write_bytes(b'not audio')
 (root/'truncated.wav').write_bytes(struct.pack('<4sI4s4sIHHIIHH4sI',b'RIFF',16036,b'WAVE',b'fmt ',16,1,1,8000,16000,2,16,b'data',16000)+b'\0'*10)
 # Complete PCM with publisher-style RIFF length or metadata errors stays readable.
 body=(root/'mono.wav').read_bytes()
 (root/'complete_bad_riff_length.wav').write_bytes(body[:4]+struct.pack('<I',len(body))+body[8:])
 tail=body+b'LIST'+struct.pack('<I',100)+b'xx'
 (root/'incomplete_metadata.wav').write_bytes(tail[:4]+struct.pack('<I',len(tail)+90)+tail[8:])
 return sorted(p for p in root.rglob('*.wav'))

def check():
 with tempfile.TemporaryDirectory(prefix='uap-any-file-') as d:
  root=Path(d)/'library';root.mkdir();paths=fixtures(root)
  result=audio_tag.library(root,None,root/'AUDIO_TAGS.csv',60,batch=4)
  with (root/'AUDIO_TAGS.csv').open(encoding='utf-8') as f:tags={r['relative_path']:r for r in csv.DictReader(f)}
  assert len(tags)==len(paths), 'Every discovered file needs an explicit tag/status row'
  for name,status in [('ten_ms.wav','too short to recognise'),('silence.wav','digital silence'),('two_hours.wav','silence in analysed excerpt'),('late_event.wav','silence in analysed excerpt'),('corrupt.wav','unreadable'),('truncated.wav','unreadable')]:
   assert tags[name].get('recognition_status')==status,(name,tags[name])
   assert not tags[name]['top_labels'] and tags[name]['clap_row']=='-1'
  for name in ('complete_bad_riff_length.wav','incomplete_metadata.wav'):
   assert tags[name]['recognition_status']=='recognised', 'Complete PCM rejected: '+name
  assert tags['opposed_5_1.wav']['recognition_status']=='recognised', 'Channel cancellation was mislabelled digital silence'
  import audio_describe
  assert audio_describe.describe_file(root/'opposed_5_1.wav')['sound_s']>.1
  index=audio_search.Index(root);found,_=index.find('sound',top=100)
  assert set(index.paths)==set(tags), 'Search index silently omitted a file'
  for path in ('ten_ms.wav','silence.wav','two_hours.wav','late_event.wav','corrupt.wav','truncated.wav'):
   found,_=index.find(Path(path).stem.replace('_',' '),top=100)
   assert path in {r['path'] for r in found}, 'Unavailable file cannot be found by its name: '+path
  lib=sound_browser.Library(root);data=lib.payload();assert len(data['rows'])==len(paths)
  searched=lib.search('corrupt');json.dumps(searched,allow_nan=False)
  assert 'corrupt.wav' in searched.get('path_scores',{}) and searched.get('scores') is None, 'Browser search dropped unreadable row'
  by={r[0]:r for r in data['rows']};assert by['corrupt.wav'][13]=='unreadable'
  assert by['two_hours.wav'][3]==7200
  empty=Path(d)/'empty';empty.mkdir();audio_tag.library(empty,None,empty/'AUDIO_TAGS.csv',60)
  assert audio_search.Index(empty).find('click')[0]==[] and sound_browser.Library(empty).payload()['rows']==[]
  # Explicit external sidecars keep source folders read-only. This tests real access denial
  # via Windows ACL when available; all source size/mtime values are also compared.
  protected=Path(d)/'readonly';protected.mkdir();sf.write(protected/'tone.wav',np.sin(np.arange(8000)*.1).astype('float32'),8000)
  state=Path(d)/'state';state.mkdir();before={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in protected.iterdir()}
  import subprocess,getpass
  account=getpass.getuser();denied=False;old_cache=os.environ.get('UAP_CACHE_ROOT');os.environ['UAP_CACHE_ROOT']=str(state)
  try:
   if os.name=='nt':
    subprocess.run(['icacls',str(protected),'/deny',account+':(OI)(CI)(WD,AD,WA,WEA)'],check=True,capture_output=True)
    denied=True
    try:(protected/'forbidden.tmp').write_bytes(b'x')
    except PermissionError:pass
    else:raise AssertionError('Read-only fixture ACL did not deny writes')
   audio_tag.library(protected,None,protected/'AUDIO_TAGS.csv',60)
   assert not (protected/'AUDIO_LEDGER_files.csv').exists(), 'External output wrote sidecars in read-only source'
   external=sound_browser.Library(protected)
   assert len(external.payload()['rows'])==1
   external.waveform('tone.wav');external.spectrogram('tone.wav');external.search('tone')
   assert before=={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in protected.iterdir()}
  finally:
   if old_cache is None:os.environ.pop('UAP_CACHE_ROOT',None)
   else:os.environ['UAP_CACHE_ROOT']=old_cache
   if denied:subprocess.run(['icacls',str(protected),'/remove:d',account],check=True,capture_output=True)
 print('PASS any-file fixtures: 10ms, silence, PCM8, float32, 8k/192k, mono/5.1 including opposing phases, 2h and late events, corrupt/truncated and complete PCM with invalid RIFF/metadata lengths, Unicode/long path, empty, external metadata for read-only sources')
if __name__=='__main__':check()
