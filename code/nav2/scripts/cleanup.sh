#!/bin/bash
# Kill every this study's Nav2 runs process from a previous run, by PID.
#
# WHY NOT pkill -x:  the loopback simulator and the delay node are Python nodes,
# so their process NAME is `python3`, not `loopback_simulator`.  `pkill -x
# loopback_simulator` silently matches nothing.  That is how three loopback
# simulators and two controller servers ended up running at once, all publishing
# to the same topics -- every measurement taken in that state is worthless.
#
# WHY NOT pkill -f <pattern>:  when the pattern is typed on the command line, the
# invoking shell's own cmdline contains it, and pkill kills the shell.
#
# So: collect PIDs with pgrep, then kill by PID, excluding this script and its
# parent.  Run this before every relaunch, and let sweep.py's assert_single_stack
# confirm the result.
set -u
ME=$$
PAR=${PPID:-0}
killed=0
for p in $(pgrep -f "jazzy/lib/nav2" 2>/dev/null) \
         $(pgrep -f "jazzy/lib/tf2_ros" 2>/dev/null) \
         $(pgrep -f "letg2_nav2_validation" 2>/dev/null) \
         $(pgrep -f "ros2 launch" 2>/dev/null); do
    # NEVER kill the experiment driver.  Its command line contains
    # "letg2_nav2_validation" (it is run as
    # src/letg2_nav2_validation/letg2_nav2_validation/experiment.py), so the third
    # pattern above matches it, and it is neither $ME nor $PAR when cleanup is
    # invoked through an intermediate `bash -c`.  Killing it would end the sweep
    # silently, mid-grid.
    if grep -qa "experiment\.py" "/proc/$p/cmdline" 2>/dev/null; then
        continue
    fi
    if [ "$p" != "$ME" ] && [ "$p" != "$PAR" ]; then
        kill -9 "$p" 2>/dev/null && killed=$((killed+1))
    fi
done
sleep 3
echo "cleanup: killed $killed process(es)"
left=$(pgrep -f "jazzy/lib/nav2" 2>/dev/null | grep -v "^$ME$" | grep -v "^$PAR$" | wc -l)
echo "cleanup: $left nav2 process(es) remain"
exit 0
