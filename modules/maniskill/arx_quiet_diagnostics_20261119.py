"""Keep fatal/on-demand traceback reporting without periodic stack walking."""
import faulthandler


def install():
    original=faulthandler.dump_traceback_later
    def dump_traceback_later(timeout,repeat=False,file=None,exit=False):
        if repeat:
            return None
        return original(timeout,repeat=repeat,file=file,exit=exit)
    faulthandler.dump_traceback_later=dump_traceback_later

