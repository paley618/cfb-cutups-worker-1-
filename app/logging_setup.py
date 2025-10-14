import json
import logging
import sys


class _SimpleJson(logging.Formatter):
    def format(self, record):
        base = {
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


def setup_logging():
    root = logging.getLogger()
    # Don’t double-add handlers on reload
    if getattr(root, "_cfb_configured", False):
        return
    root.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    # Plain text is fine; if you want JSON, swap the formatter:
    # h.setFormatter(_SimpleJson())
    fmt = logging.Formatter("%(levelname)s %(name)s | %(message)s")
    h.setFormatter(fmt)
    root.addHandler(h)
    root._cfb_configured = True
