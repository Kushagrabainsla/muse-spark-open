#!/bin/bash
# nsk.sh args...: kernel time table (avg us, name) for prof2.py args
source /root/.okcenv; cd /root/okc_attn
nsys profile -c cudaProfilerApi -o /tmp/p_$$ -f true python prof2.py "$@" >/dev/null 2>&1
nsys stats -r cuda_gpu_kern_sum -f csv /tmp/p_$$.nsys-rep 2>/dev/null | python -c "
import csv,sys
r=[x for x in csv.reader(sys.stdin) if len(x)>5]
h=r[0]; ia=h.index('Avg (ns)'); iname=h.index('Name')
for row in r[1:]: print(f'{float(row[ia])/1e3:9.1f} us  {row[iname][:60]}')"
