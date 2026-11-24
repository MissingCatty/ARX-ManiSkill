"""Small process-independent interfaces for interchangeable pipeline modules."""
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any,Protocol


class Worker(Protocol):
    def request(self,directory:Path,**options:Any)->dict[str,Any]:...


@dataclass(frozen=True)
class Stage:
    id:str
    module:str
    required_files:tuple[str,...]=()
    produced_files:tuple[str,...]=()
    options:dict[str,Any]=field(default_factory=dict)


@dataclass(frozen=True)
class StageOutcome:
    stage:str
    status:str
    elapsed_seconds:float
    details:dict[str,Any]
