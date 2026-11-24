"""Install the official rclone binary with its published SHA256 checksum."""
import hashlib
import concurrent.futures
import argparse
import shutil
import subprocess
import tarfile
import re
import time
import urllib.request
import zipfile

from arx_project.layout import ROOT, write_json, now


def fetch(url, destination=None):
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=40) as response:
                if destination is None:
                    return response.read()
                with destination.open('wb') as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def install_official():
    version = re.search(r'v\d+\.\d+\.\d+', fetch('https://downloads.rclone.org/version.txt').decode()).group()
    name = f'rclone-{version}-linux-amd64.zip'
    base = f'https://downloads.rclone.org/{version}/'
    checks = fetch(base + 'SHA256SUMS').decode()
    expected = next(parts[0] for line in checks.splitlines()
                    if len(parts := line.split()) == 2 and parts[1].lstrip('*') == name)
    work = ROOT / 'tmp/cloud_backup/tool_install'
    work.mkdir(parents=True, exist_ok=True)
    archive = work / name
    print('Downloading official rclone', version, flush=True)
    request = urllib.request.Request(base + name + '?size=1', headers={
        'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-0'})
    with urllib.request.urlopen(request, timeout=40) as response:
        total = int(response.headers['Content-Range'].split('/')[-1])
    parts = work / (name + '.parts')
    parts.mkdir(exist_ok=True)
    chunk_size = 4 * 1024 * 1024

    def download_part(index):
        start, end = index * chunk_size, min(total, (index + 1) * chunk_size) - 1
        target = parts / f'{index:05d}'
        if target.exists() and target.stat().st_size == end - start + 1:
            return target
        for attempt in range(4):
            try:
                request = urllib.request.Request(base + name + f'?range_start={start}', headers={
                    'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=40) as response:
                    if response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                        raise ValueError('Unexpected download byte range')
                    data = response.read()
                if len(data) != end - start + 1:
                    raise ValueError('Truncated rclone download part')
                target.write_bytes(data)
                return target
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        downloads = list(pool.map(download_part, range((total + chunk_size - 1) // chunk_size)))
    with archive.open('wb') as output:
        for part in downloads:
            output.write(part.read_bytes())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError('Official rclone archive checksum mismatch')
    binary = ROOT / 'environments/backup_tools/bin/rclone'
    binary.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        binary.write_bytes(bundle.read(f'rclone-{version}-linux-amd64/rclone'))
    binary.chmod(0o755)
    write_json(ROOT / 'configs/environments/backup_tools.json', {
        'name': 'backup_tools', 'type': 'native_tools', 'installed_at': now(),
        'tools': {'rclone': {'version': version, 'source': base + name,
                            'archive_sha256': digest,
                            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}}})
    print('Installed', version, 'with verified SHA256', flush=True)


def install_ubuntu():
    metadata = subprocess.check_output(['apt-cache', 'show', '--no-all-versions', 'rclone'], text=True)
    fields = dict(line.split(': ', 1) for line in metadata.splitlines() if ': ' in line and not line.startswith(' '))
    work = ROOT / 'tmp/cloud_backup/tool_install'
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run(['apt-get', 'download', 'rclone=' + fields['Version']], cwd=work, check=True)
    archive = next(p for p in work.glob('rclone_*.deb') if subprocess.check_output(
        ['dpkg-deb', '--field', str(p), 'Version'], text=True).strip() == fields['Version'])
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != fields['SHA256']:
        raise ValueError('Ubuntu rclone package checksum mismatch')
    process = subprocess.Popen(['dpkg-deb', '--fsys-tarfile', str(archive)], stdout=subprocess.PIPE)
    binary = ROOT / 'environments/backup_tools/bin/rclone'
    binary.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
        for member in stream:
            if member.name == './usr/bin/rclone':
                with stream.extractfile(member) as source, binary.open('wb') as output:
                    shutil.copyfileobj(source, output)
    if process.wait():
        raise RuntimeError('Could not extract the Ubuntu rclone package')
    binary.chmod(0o755)
    write_json(ROOT / 'configs/environments/backup_tools.json', {
        'name': 'backup_tools', 'type': 'native_tools', 'installed_at': now(),
        'tools': {'rclone': {'version': fields['Version'], 'source': fields['Filename'],
                            'provider': 'Ubuntu package repository', 'archive_sha256': digest,
                            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}}})
    print('Installed Ubuntu rclone', fields['Version'], 'with verified SHA256', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-release', action='store_true')
    args = parser.parse_args()
    if not args.official_release and all(shutil.which(name) for name in ['apt-get', 'apt-cache', 'dpkg-deb']):
        install_ubuntu()
    else:
        install_official()
