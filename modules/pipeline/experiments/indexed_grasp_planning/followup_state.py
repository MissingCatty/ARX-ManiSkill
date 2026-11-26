"""Confirm completion from a fresh receipt after observing controller exit."""


def completed_controller(fields, read_latest, expected_requests):
    required = {'MainPID', 'ActiveState', 'ExecMainStatus'}
    if not required.issubset(fields):
        raise RuntimeError('Incomplete controller exit status')
    if fields['MainPID'] != '0' or fields['ActiveState'] in ['active', 'activating', 'reloading', 'deactivating']:
        return None
    # A previously read progress snapshot can precede the last write even
    # when systemd now reports a successful exit. Read in this order.
    progress = read_latest()
    if fields['ActiveState'] != 'inactive' or fields['ExecMainStatus'] != '0':
        raise RuntimeError('Validation controller did not exit successfully')
    completed, total = progress.get('completed_requests'), progress.get('total_requests')
    if progress.get('phase') != 'complete' or type(completed) is not int or type(total) is not int \
            or completed != total or total != expected_requests or total <= 0:
        raise RuntimeError('Validation controller exited before completing registered requests')
    return progress
