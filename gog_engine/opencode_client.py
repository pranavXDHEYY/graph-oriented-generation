import os
import subprocess
import shutil

# TensorFlow and CUDA print noisy warnings to stderr even when not used.
# These originate from opencode's subprocess environment, not our code.
# We suppress them by filtering known benign prefixes from stderr output.
_STDERR_NOISE_PREFIXES = (
    "E external/local_xla",
    "WARNING: All log messages",
    "E0000",
    "W0000",
    "I tensorflow",
    "AttributeError: 'MessageFactory'",
    "A module that was compiled using NumPy 1.x",
    "NumPy 2.",
    "Some module may need",
    "If you are a user",
    "We expect that some",
    "huggingface/tokenizers:",
    "To disable this warning",
    "- Avoid using",
    "- Explicitly set",
)


def _is_noise(line: str) -> bool:
    return any(line.strip().startswith(p) for p in _STDERR_NOISE_PREFIXES)


class OpenCodeClient:
    """Connector for the OpenCode CLI using the @mention context system."""

    def __init__(self, binary="opencode"):
        self.binary = binary
        self.is_present = shutil.which(binary) is not None

    def complete(self, prompt, context_files=None):
        """Sends a prompt with @mention file context to the OpenCode CLI."""
        if not self.is_present:
            return f"[Error] '{self.binary}' CLI not found in PATH."

        cmd = [self.binary, "run", prompt]

        if context_files:
            for file in context_files:
                if os.path.exists(file):
                    cmd.append(f"@{file}")

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if process.returncode != 0:
                # Filter noise from stderr before surfacing real errors
                real_errors = [
                    l for l in process.stderr.splitlines() if not _is_noise(l)
                ]
                return f"[Error calling '{self.binary}'] {chr(10).join(real_errors).strip()}"

            return process.stdout.strip()
        except Exception as e:
            return f"[Error calling '{self.binary}'] {e}"


if __name__ == "__main__":
    client = OpenCodeClient()
    if client.is_present:
        print("Testing OpenCode CLI...")
        print(client.complete("Explain this project.", context_files=["README.md"]))
    else:
        print("OpenCode CLI not found.")