"""Library metadata shared by tagging and readers; existing ledger annotations are retained."""
import csv
import math
import os
import sys
import tempfile
from pathlib import Path

AUDIO_EXTENSIONS = {'.wav','.mp3','.ogg','.flac','.aif','.aiff','.m4a','.aac','.opus','.mp4','.mov'}
LEDGER_NAME = 'AUDIO_LEDGER_files.csv'
LEDGER_FIELDS = ['relative_path', 'pack', 'duration_seconds', 'sample_rate', 'channels', 'size_bytes', 'mtime_ns']
_missing_ledger_noted = set()


def metadata_directory(root, explicit=None, create=False):
    """Use external sidecars automatically when a library refuses writes."""
    import hashlib
    root=Path(root).resolve()
    if explicit is not None:return Path(explicit).resolve()
    base=Path(os.environ.get('UAP_CACHE_ROOT') or
              Path(os.environ.get('LOCALAPPDATA',Path.home()/'.cache'))/'UltimateAudioPartner')
    cache=base/'libraries'/hashlib.sha256(str(root).casefold().encode('utf-8')).hexdigest()[:24]
    if (cache/'AUDIO_TAGS.csv').exists():return cache
    if not create:return root
    try:
        import uuid
        probe=root/('.uap-write-'+uuid.uuid4().hex)
        descriptor=os.open(probe,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        os.close(descriptor);probe.unlink()
        return root
    except PermissionError:
        cache.mkdir(parents=True,exist_ok=True)
        print(f'NOTE library is read-only; metadata will be saved in {cache}',file=sys.stderr,flush=True)
        return cache


def relative_path(value):
    return str(value).replace('\\', '/')


def pack_from_path(value):
    parts = relative_path(value).split('/')
    return parts[0] if len(parts) > 1 else ''


def note_missing_ledger(root):
    key = str(Path(root).resolve())
    if key not in _missing_ledger_noted:
        print(f'NOTE {LEDGER_NAME} missing; using AUDIO_TAGS.csv (durations cover only the tagged span).',
              file=sys.stderr, flush=True)
        _missing_ledger_noted.add(key)


def tag_metadata(tags):
    """AUDIO_TAGS duration is an analysed span, and therefore a lower bound on full duration."""
    return {relative_path(path): {**row, 'relative_path': relative_path(path),
            'pack': row.get('pack') or pack_from_path(path),
            'duration_seconds': row.get('duration_seconds') or row.get('seconds_tagged') or ''}
            for path, row in tags.items()}


def read_metadata(root, tags, metadata_root=None):
    """Merge ledger rows over tag metadata. A missing ledger is reported once per root/process."""
    result = tag_metadata(tags)
    ledger = Path(metadata_root or root) / LEDGER_NAME
    if not ledger.exists():
        note_missing_ledger(root)
        for path in Path(root).rglob('*'):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:continue
            rel=path.relative_to(root).as_posix()
            if rel in result:continue
            row={'relative_path':rel,'pack':pack_from_path(rel),'recognition_status':'not tagged'}
            try:row.update(measure_metadata(path))
            except Exception as exc:row.update(recognition_status='unreadable',error=str(exc))
            result[rel]=row
        return result
    with ledger.open(encoding='utf-8-sig', newline='') as source:
        for row in csv.DictReader(source):
            path = relative_path(row['relative_path'])
            result[path] = {**result.get(path, {'relative_path': path, 'pack': pack_from_path(path)}),
                            **{key: value for key, value in row.items() if value not in ('', None)},
                            'relative_path': path}
    # Older passes left unreadable files only in the ledger. Probe untagged rows with
    # no duration so those failures are explicit without changing protected sidecars.
    for rel,row in result.items():
        if rel not in tags and not row.get('duration_seconds') and not row.get('recognition_status'):
            try:
                from audio_map import load_audio
                samples,_=load_audio(Path(root)/rel,seconds=.05,sample_rate=16000,mono=True)
                if not len(samples):raise ValueError('unreadable: no decoded samples')
                row.update(measure_metadata(Path(root)/rel),recognition_status='not tagged')
            except Exception as exc:row.update(recognition_status='unreadable',error=str(exc))
    skipped=Path(metadata_root or root)/'AUDIO_TAGS_skipped.csv'
    if skipped.exists():
        with skipped.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                rel=relative_path(row['relative_path']);result.setdefault(rel,{'relative_path':rel,'pack':pack_from_path(rel)})
                result[rel].update(recognition_status='unreadable',error=row.get('reason',''))
    return result


def write_csv_atomic(path, fields, rows):
    """Write in the destination folder, then replace; readers see either complete snapshot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', prefix=path.name + '.',
                                         suffix='.tmp', dir=path.parent, delete=False) as target:
            temporary = Path(target.name)
            writer = csv.DictWriter(target, fields)
            writer.writeheader()
            writer.writerows(rows)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def measure_metadata(path):
    """Probe full container duration/channel count; recognition still uses its capped excerpt."""
    import soundfile as sf
    try:
        info = sf.info(path)
        duration, sample_rate, channels = info.duration, info.samplerate, info.channels
    except (sf.LibsndfileError, ValueError):
        from audio_map import probe
        info = probe(path)
        stream = next(s for s in info['streams'] if s.get('codec_type') == 'audio')
        duration = float(stream.get('duration') or info.get('format', {}).get('duration') or 0)
        sample_rate, channels = int(stream['sample_rate']), int(stream['channels'])
    if not math.isfinite(duration) or duration < 0 or sample_rate <= 0 or channels <= 0:
        raise ValueError('Invalid audio duration, sample rate or channels')
    return {'duration_seconds': round(duration, 6), 'sample_rate': sample_rate, 'channels': channels}


def refresh_ledger(root, files, metadata_root=None):
    """Refresh only new/changed files, preserving all old columns and rows, including missing files."""
    from concurrent.futures import ThreadPoolExecutor
    root = Path(root)
    ledger = Path(metadata_root or root) / LEDGER_NAME
    rows, fields = [], []
    if ledger.exists():
        with ledger.open(encoding='utf-8-sig', newline='') as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or [])
            if 'relative_path' not in fields:
                raise ValueError(f'{LEDGER_NAME} needs a relative_path column')
            rows = list(reader)
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                raise ValueError(f'{LEDGER_NAME} has malformed rows; refusing to overwrite it')
    original_fields = list(fields)
    fields.extend(field for field in LEDGER_FIELDS if field not in fields)
    owners = {relative_path(row['relative_path']): i for i, row in enumerate(rows)}
    pending, backfilled = [], False
    for path in files:
        path = Path(path)
        rel, stat = path.relative_to(root).as_posix(), path.stat()
        index = owners.get(rel)
        old = rows[index] if index is not None else {}
        measured = all(old.get(key) not in ('', None) for key in ('duration_seconds', 'sample_rate', 'channels'))
        if measured and str(old.get('size_bytes', '')) == str(stat.st_size):
            if str(old.get('mtime_ns', '')) == str(stat.st_mtime_ns):
                continue
            if old.get('mtime_ns') in ('', None):  # a ledger from before mtime_ns: backfill it, do not re-probe
                old['mtime_ns'] = stat.st_mtime_ns
                backfilled = True
                continue
        pending.append((path, rel, stat, index))
    def attempt(item):
        path, rel, stat, index = item
        values = {'relative_path': rel, 'pack': pack_from_path(rel), 'size_bytes': stat.st_size,
                  'mtime_ns': stat.st_mtime_ns, 'duration_seconds': '', 'sample_rate': '', 'channels': ''}
        try:
            values.update(measure_metadata(path))
            return index, values, None
        except Exception as exc:
            return index, values, str(exc)
    errors = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, values, error in pool.map(attempt, pending):
            if index is None:
                rows.append({field: '' for field in fields} | values)
            else:
                if rows[index].get('pack'):  # an existing ledger's pack may be finer than the first folder: keep it
                    values.pop('pack')
                rows[index].update(values)
            if error:
                errors[values['relative_path']] = error
    if not ledger.exists() or fields != original_fields or pending or backfilled:
        write_csv_atomic(ledger, fields, rows)
    return errors
