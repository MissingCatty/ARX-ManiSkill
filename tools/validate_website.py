"""Test a copied website without its experiment modules, assets or output tree."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import os

ROOT=Path(__file__).resolve().parents[1]


def main():
    website=ROOT/'website'
    node=shutil.which('node')
    if not node:raise SystemExit('Node.js 22.13+ is required')
    base=ROOT/'tmp/website';base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='standalone_',dir=base) as folder:
        target=Path(folder)/'website'
        def ignore(directory,names):
            if Path(directory)==website/'resources/state':
                return [name for name in names if name!='todos.snapshot.json']
            return [n for n in names if n in {'dist','data','数据'}] if Path(directory)==website else []
        shutil.copytree(website,target,symlinks=False,ignore=ignore)
        # The copy contains no modules/, simulation/ or project tmp/ sibling.
        build=subprocess.run([node,'build.mjs'],cwd=target,capture_output=True,text=True,timeout=120)
        if build.returncode:raise RuntimeError(build.stderr)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        log=(Path(folder)/'server.log').open('w')
        server=subprocess.Popen([node,'server.mjs'],cwd=target,env={**os.environ,'CLARO_PORT':str(port),'CLARO_MDNS_NAME':'arx-snapshot-test.local'},stdout=log,stderr=log)
        try:
            start=time.monotonic()
            while True:
                try:
                    with urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as response:
                        if response.status==200:break
                except OSError:
                    if server.poll() is not None or time.monotonic()-start>20:raise RuntimeError('Standalone server did not start')
                    time.sleep(.1)
            files=[];site=target/'resources/site'
            for file in site.rglob('*'):
                if not file.is_file() or file.suffix in ['.gz','.br']:continue
                path='/'+str(file.relative_to(site))
                with urlopen(Request(f'http://127.0.0.1:{port}'+__import__('urllib.parse',fromlist=['quote']).quote(path),method='HEAD'),timeout=5) as response:
                    if response.status!=200:raise RuntimeError('Missing resource: '+path)
                files.append(path)
            video=next(site.glob('media/*.mp4'))
            with urlopen(Request(f'http://127.0.0.1:{port}/media/'+video.name,headers={'Range':'bytes=0-1023'}),timeout=5) as response:
                assert response.status==206 and len(response.read())==1024
            with urlopen(f'http://127.0.0.1:{port}/api/todos',timeout=5) as response:assert response.status==200
            snapshot_file=target/'resources/state/todos.snapshot.json'
            snapshot=json.loads(snapshot_file.read_text()) if snapshot_file.exists() else None
            if snapshot:
                for version in {item['version'] for item in snapshot['items']}:
                    url=f'http://127.0.0.1:{port}/api/todos?version='+__import__('urllib.parse',fromlist=['quote']).quote(version)
                    with urlopen(url,timeout=5) as response:
                        actual=json.loads(response.read())
                    expected=[item for item in snapshot['items'] if item['version']==version]
                    assert sorted(actual,key=lambda row:row['id'])==sorted(expected,key=lambda row:row['id'])
            # Verify encoded representations, explicit opt-outs and the image format.
            for accepted, expected in [('br, gzip','br'),('br;q=0, gzip','gzip'),('br;q=0, gzip;q=0',None)]:
                with urlopen(Request(f'http://127.0.0.1:{port}/',headers={'Accept-Encoding':accepted}),timeout=5) as response:
                    assert response.headers.get('Content-Encoding')==expected
                    assert response.headers.get('Vary')=='Accept-Encoding'
                    payload=response.read()
                    want=(site/('index.html'+({None:'','br':'.br','gzip':'.gz'}[expected]))).read_bytes()
                    assert payload==want
            webp=next(site.glob('optimized/media/*.webp'))
            with urlopen(f'http://127.0.0.1:{port}/'+str(webp.relative_to(site)),timeout=5) as response:
                assert response.headers.get('Content-Type')=='image/webp'
                assert 'immutable' in response.headers.get('Cache-Control','')
                assert response.read(4)==b'RIFF'
            for file in (target/'resources').rglob('*'):
                if file.is_symlink():raise RuntimeError('Published resource is a symlink: '+str(file))
            result={'passed':True,'standalone_rebuild':True,'standalone_server':True,'files_http_checked':len(files),'media_files':sum(p.startswith('/media/') for p in files),'video_range':True,'todo_api':True,'todo_snapshot_restore':snapshot is not None,'todo_snapshot_items':len(snapshot['items']) if snapshot else 0,'webp':True,'brotli_gzip_and_opt_out':True,'experiment_directories_present':False}
            (base/'standalone_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()
            log.close()


if __name__=='__main__':main()
