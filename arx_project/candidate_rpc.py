"""Candidate handoff lifecycle around the unchanged small-message RPC wire."""
from pathlib import Path

from .artifacts import (FORMAT, adopt_candidate_transport, candidate_transport,
                        finish_candidate_transport)
from .registry import module_records


def call_with_candidate_transport(client, message, call):
    command = message.get('command', 'request')
    directory = message.get('directory')
    descriptor = None
    outgoing = dict(message)
    if command == 'request' and directory is not None:
        record = module_records().get(Path(client.path).stem, {})
        supported = getattr(client, 'accepts_candidate_transport', None)
        if supported is None:
            supported = record.get('candidate_transport_format') == FORMAT
        if record.get('role') == 'motion_planning' or supported:
            descriptor = candidate_transport(directory)
            if descriptor is not None:
                if supported:
                    if 'candidate_transport' in outgoing and outgoing['candidate_transport'] != descriptor:
                        raise ValueError('Conflicting candidate handoff for this observation')
                    outgoing['candidate_transport'] = descriptor
                else:
                    # Existing single-arm/file consumers retain their input contract.
                    finish_candidate_transport(directory, client_factory=type(client))
                    descriptor = None
    try:
        response = call(outgoing)
    finally:
        if descriptor is not None:
            # Join the producer's background save after the downstream computation.
            finish_candidate_transport(directory, client_factory=type(client))
    transport = response.get('result', {}).get('candidate_transport')
    if transport is not None:
        adopt_candidate_transport(directory, transport, client.path)
        if transport['count'] == 0:
            finish_candidate_transport(directory, client_factory=type(client))
    return response
