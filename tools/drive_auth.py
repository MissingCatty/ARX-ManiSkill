"""Browser OAuth for the configured Google Drive account; tokens stay outside Git."""
import configparser
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.parse
import urllib.request

from arx_project.layout import ROOT, read_json, write_json, now


def main():
    cfg = read_json(ROOT / 'configs/cloud_backup.json')
    private = Path(cfg['credential_file']).expanduser()
    private.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output = ROOT / 'tmp/cloud_backup/auth'
    output.mkdir(parents=True, exist_ok=True)
    status = output / 'status.json'
    base = {'at': now(), 'account_email': cfg['account_email'],
            'folder_id': cfg['folder_id'], 'pid': os.getpid()}
    write_json(status, {**base, 'stage': 'starting_authorization'})
    raw_log = private.parent / 'google-drive-auth.log'
    command = [str(ROOT / cfg['rclone_binary']), 'authorize', 'drive', '--auth-no-open-browser']
    client_file = private.parent / 'google-oauth-client.json'
    if client_file.exists():
        client = json.loads(client_file.read_text())
        client = client.get('installed', client)
        # Client credentials travel to the child through its environment,
        # rather than command arguments or the project logs.
        environment = {**os.environ, 'RCLONE_DRIVE_CLIENT_ID': client['client_id'],
                       'RCLONE_DRIVE_CLIENT_SECRET': client['client_secret']}
    else:
        environment = os.environ.copy()
    fd = os.open(raw_log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, 'w') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=environment)
        for line in process.stdout:
            log.write(line)
            log.flush()
            if match := re.search(r'http://127\.0\.0\.1:\d+/auth\?[^\s]+', line):
                write_json(status, {**base, 'stage': 'waiting_for_browser', 'url': match.group()})
        code = process.wait()
    if code:
        raise RuntimeError('rclone authorization failed; private log: ' + str(raw_log))
    text = raw_log.read_text()
    token = None
    for match in re.finditer(r'\{\s*"access_token"', text):
        try:
            token, _ = json.JSONDecoder().raw_decode(text[match.start():])
            break
        except ValueError:
            pass
    if not token:
        raise RuntimeError('No Google OAuth token was returned')

    def api(path, fields):
        url = 'https://www.googleapis.com/drive/v3/' + path + '?' + urllib.parse.urlencode({
            'fields': fields, 'supportsAllDrives': 'true'} if path.startswith('files/') else {'fields': fields})
        request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token['access_token'],
                                                      'User-Agent': 'ARX-ManiSkill-backup'})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    about = api('about', 'user(emailAddress),storageQuota')
    email = about['user']['emailAddress']
    if email.lower() != cfg['account_email'].lower():
        raise ValueError('Expected Google account ' + cfg['account_email'] + '; authenticated account: ' + email)
    folder = api('files/' + cfg['folder_id'], 'id,name,mimeType,trashed,capabilities(canAddChildren)')
    if folder['mimeType'] != 'application/vnd.google-apps.folder' or folder.get('trashed') or not folder['capabilities'].get('canAddChildren'):
        raise ValueError('The configured Google Drive folder is not writable')
    parser = configparser.RawConfigParser()
    parser[cfg['remote_name']] = {'type': 'drive', 'scope': 'drive', 'root_folder_id': cfg['folder_id'],
                                  'token': json.dumps(token)}
    if client_file.exists():
        parser[cfg['remote_name']].update(client_id=client['client_id'], client_secret=client['client_secret'])
    fd = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, 'w') as handle:
        parser.write(handle)
    write_json(status, {**base, 'stage': 'authorized', 'folder_name': folder['name'],
                        'storage_quota': about['storageQuota']})
    raw_log.unlink()
    print('Google Drive authorized for', email, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_json(ROOT / 'tmp/cloud_backup/auth/status.json', {
            'stage': 'authorization_failed', 'at': now(), 'error': str(error)})
        raise
