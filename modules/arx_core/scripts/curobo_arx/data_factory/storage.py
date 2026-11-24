"""Declared overflow storage for long collections; preserve local metadata."""
from pathlib import Path
from .common import space,write


def register_bulk_storage(campaign,root):
    campaign=Path(campaign).resolve();root=Path(root)
    if not root.is_absolute() or not root.is_relative_to('/coolas/data'):
        raise ValueError('Overflow storage must be a dedicated absolute /coolas/data directory')
    space('/coolas/data')
    root.mkdir(parents=True,exist_ok=False)
    for name in ('groups','workers'):
        target=root/name;target.mkdir()
        (campaign/name).symlink_to(target,target_is_directory=True)
    value={'root':str(root.resolve()),'directories':['groups','workers'],
        'metadata_root':str(campaign),'reason':'Long collection does not fit available local space; preserve the 1 GiB reserve on both filesystems',
        'old_files_moved_or_deleted':False}
    write(campaign/'bulk_storage.json',value)
    return value


def container_storage_mounts(campaign,registration):
    value=registration.get('bulk_storage')
    if not value:return []
    root=Path(value['root']);campaign=Path(campaign)
    if not root.is_absolute() or not root.is_relative_to('/coolas/data') or not root.is_dir():
        raise ValueError('Registered overflow storage is unavailable')
    if value['directories']!=['groups','workers']:
        raise ValueError('Unknown bulk storage directory mapping')
    for name in value['directories']:
        path=campaign/name
        if not path.is_symlink() or path.resolve()!=(root/name).resolve():
            raise ValueError('Overflow symlink differs from registration: '+str(path))
    space(root)
    return ['-v',str(root)+':'+str(root)+':rw']
