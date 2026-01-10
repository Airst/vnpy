
import re
from datetime import datetime

# --- Logger Redirection ---
class LoggerWriter:
    def __init__(self, writer, file):
        self.writer = writer
        self.file = file
        self.last_char = "\n"

    def write(self, message):
        if not message:
            return
        
        # Strip ANSI escape codes for file writing and timestamp checking
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        clean_message = ansi_escape.sub('', message)

        # Check if the message already has a timestamp (YYYY-MM-DD)
        has_timestamp = re.search(r'^\s*(\[)?\d{4}-\d{2}-\d{2}', clean_message)

        # Determine if we need to add a timestamp
        prefix = ""
        if (self.last_char == "\n" or self.last_char == "\r") and not has_timestamp and clean_message.strip() != "":
            prefix = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")

        # Write to file (clean version)
        if self.file:
            try:
                self.file.write(prefix + clean_message)
                self.file.flush()
            except Exception:
                pass

        # Write to console (original version with colors)
        # if self.writer:
        #     try:
        #         self.writer.write(prefix + message)
        #         if hasattr(self.writer, "flush"):
        #             self.writer.flush()
        #     except Exception:
        #         pass
        
        if message:
            self.last_char = message[-1]

    def flush(self):
        if hasattr(self.writer, "flush"):
            self.writer.flush()
        if self.file:
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()

    def isatty(self):
        if hasattr(self.writer, "isatty"):
            return self.writer.isatty()
        return False

    def fileno(self):
        if hasattr(self.writer, "fileno"):
            return self.writer.fileno()
        raise OSError("LoggerWriter has no fileno")