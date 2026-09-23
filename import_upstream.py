"""Import the pinned CuPBoP_Vortex core; do not run over local adaptations."""
from pathlib import Path
import hashlib,json,subprocess
C=Path(__file__).resolve().parent
U=C
PIN='ee05a48d79cbae11351e9c8eb711f286c1206402'
subprocess.run(['git','-C',str(U),'cat-file','-e',PIN+'^{commit}'],check=True)
paths=['include/'+n+'.h' for n in ['insert_warp_loop','insert_sync','handle_sync','cg_sync','tool']]
paths+=['src/'+n+'.cpp' for n in ['insert_warp_loop','insert_sync','handle_sync','cg_sync']]
manifest=[]
for relative in paths:
    target=C/'cupbop'/relative
    assert not target.exists(), 'Refusing to overwrite adapted source: '+str(target)
    upstream='compilation/KernelTranslation/'+relative
    data=subprocess.check_output(['git','-C',str(U),'show',PIN+':'+upstream])
    source=data.decode()
    if relative=='src/insert_warp_loop.cpp':
        source=source.replace('if (g_schedule_flag == 0 && need_nested_loop)',
                              'if (!cupbop_cpu_per_block_context() && g_schedule_flag == 0 && need_nested_loop)')
        source=source.replace('    if (need_nested_loop) {\n      if (const LoadInst *LI',
                              '    { // CPU lane-index loads are divergent in flat and nested loops.\n      if (const LoadInst *LI')
    if relative=='include/tool.h':
        source=source.replace('#include <cstdlib>','#include <cstdlib>\n// CPU ownership: each invocation handles one active block.\ninline bool cupbop_cpu_per_block_context() { return true; }')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('// Derived from CuPBoP_Vortex '+PIN+'; see ../LICENSE and provenance.json.\n'+source)
    manifest.append({'source':upstream,'upstream_sha256':hashlib.sha256(data).hexdigest(),'local':str(target.relative_to(C))})
(C/'cupbop/LICENSE').write_bytes(subprocess.check_output(['git','-C',str(U),'show',PIN+':LICENSE']))
(C/'cupbop/provenance.json').write_text(json.dumps({'repository':'https://github.com/cupbop/CuPBoP_Vortex','commit':PIN,'files':manifest},indent=2)+'\n')
