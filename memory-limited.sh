#!/bin/bash
ulimit -Sv 14000000 # memory limit in kB to avoid crashing
"$@"
