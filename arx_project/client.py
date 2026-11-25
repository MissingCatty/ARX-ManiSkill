"""Local RPC client shared by orchestration and standalone module callers."""
import json
import socket


class Client:
    def __init__(self,path,timeout=900,*,candidate_transport=None):
        self.path=str(path);self.timeout=timeout
        self.accepts_candidate_transport=candidate_transport

    def call(self,message):
        from .candidate_rpc import call_with_candidate_transport
        return call_with_candidate_transport(self, message, self._call)

    def _call(self,message):
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout);connection.connect(self.path)
            with connection.makefile('rwb') as stream:
                stream.write(json.dumps(message).encode()+b'\n');stream.flush()
                line=stream.readline(1024*1024)
                if not line or len(line)>=1024*1024:raise RuntimeError('Invalid RPC response')
                return json.loads(line)

    def request(self,directory,**options):return self.call({'directory':str(directory),**options})
