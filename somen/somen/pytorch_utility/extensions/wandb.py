from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import wandb
from ignite.engine import Engine, Events

from somen.pytorch_utility.extensions.extension import PRIORITY_READER, Extension
from somen.pytorch_utility.extensions.get_attached_extension import get_attached_extension
from somen.pytorch_utility.extensions.log_report import LogReport
from somen.types import PathLike


def wandb_init(out: PathLike, resume_from_id: bool = False, **kwargs) -> None:
    run_id_path = Path(out) / "wandb_run_id"

    if resume_from_id and run_id_path.exists():
        # resume
        with run_id_path.open("r") as fp:
            run_id = fp.read().strip()
    else:
        # new run
        run_id = wandb.util.generate_id()
        with run_id_path.open("w") as fp:
            fp.write(run_id)

    wandb.init(resume=run_id, **kwargs)


class WandbReporter(Extension):

    priority = PRIORITY_READER
    main_process_only = True

    def __init__(self, entries: Optional[Sequence[str]] = None, log_report: Union[str, LogReport] = "LogReport"):
        self.entries = entries
        self.log_report = log_report
        self.call_event = Events.ITERATION_COMPLETED
        self._step = 0

    def started(self, engine: Engine) -> None:
        if isinstance(self.log_report, str):
            ext = get_attached_extension(engine, self.log_report)
            if not isinstance(ext, LogReport):
                raise ValueError("Referenced extension must be an instance of `LogReport`.")
            self.log_report = ext

        assert isinstance(self.log_report, LogReport)

    def __call__(self, engine: Engine) -> None:
        if not isinstance(self.log_report, LogReport):
            raise ValueError("Please call started() before calling")

        for log in self.log_report.logs[self._step :]:
            keys = log.keys() if self.entries is None else self.entries
            wandb.log({key: log[key] for key in keys if key in log})
            self._step += 1

    def state_dict(self) -> Dict:
        return {"_step": self._step}

    def load_state_dict(self, to_load: Mapping) -> None:
        self._step = to_load["_step"]
