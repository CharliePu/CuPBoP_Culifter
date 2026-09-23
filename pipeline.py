"""Active locally adapted CuPBoP block/lane/warp CPU lowering.

Generalized per-region grid/cluster scheduling is not implemented.
Preserve inputs and record every artifact; see common/cpu-coarsening/README.md.
"""
import argparse,hashlib,json,os,re,subprocess,tempfile,time,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LOCK=Path('/tmp/culifter-benchmark.lock')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def compiler_sources():
    paths=[HERE/'build.sh']
    for folder,pattern in [('src','*.cpp'),('src','*.h'),('cupbop/src','*.cpp'),('cupbop/include','*.h')]:
        paths.extend((HERE/folder).glob(pattern))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
def descendant(pid):
    current=os.getpid()
    while current>1:
        if current==pid:return True
        try:current=int(Path(f'/proc/{current}/stat').read_text().rsplit(')',1)[1].split()[1])
        except (OSError,ValueError):return False
    return False
class CpuLock:
    def __enter__(self):
        self.token=None
        try:LOCK.mkdir()
        except FileExistsError:
            owner=(LOCK/'owner').read_text() if (LOCK/'owner').exists() else ''
            match=re.search(r'^pid=(\d+)$',owner,re.M)
            if not match or not descendant(int(match[1])):raise RuntimeError('CPU lock held by another task: '+owner)
            return self
        self.token=uuid.uuid4().hex
        (LOCK/'token').write_text(self.token)
        (LOCK/'owner').write_text(f'owner=cpu-coarsening\npid={os.getpid()}\npurpose=compile/validate lifted CPU IR\n')
        return self
    def __exit__(self,*_):
        if self.token and (LOCK/'token').read_text().strip()==self.token:
            (LOCK/'owner').unlink();(LOCK/'token').unlink();LOCK.rmdir()
def command(args,timeout=180):
    p=subprocess.run(list(map(str,args)),text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError((p.stderr[:4000]+'\n...\n'+p.stderr[-2000:]) if p.stderr else p.stdout[-6000:] or f'command exited {p.returncode}')
    return p
def compile_kernel(source,kernel,output,block_size=32,target='x86_64-linux-gnu',rename=None,obj=None,shared_memory_bytes=32768):
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output:raise ValueError('Input IR must remain unchanged')
    if target not in ['x86_64-linux-gnu','aarch64-linux-gnu']:raise ValueError('Unsupported CPU target')
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest=output.with_suffix('.json')
    with CpuLock():
        # Incremental build also detects edits to the adapted upstream pass.
        command(['bash',HERE/'build.sh'])
        source_identity=compiler_sources()
        binary_identity=sha(HERE/'build/cpu-coarsen')
        with tempfile.TemporaryDirectory(prefix='.coarsen-',dir=output.parent) as temp:
            temp=Path(temp);transformed=temp/'kernel.ll'
            args=[HERE/'build/cpu-coarsen',source,'--kernel',kernel,'--block-size',block_size,'--shared-memory-bytes',shared_memory_bytes,'--target',target,'-o',transformed]
            if rename:args+=['--rename',rename]
            try:
                result=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=180)
                output.with_suffix('.compiler.log').write_text(result.stderr)
                if result.returncode: raise RuntimeError(result.stderr[:3500]+'\n'+result.stderr[-2000:])
                command(['opt-18','-passes=verify','-disable-output',transformed],30)
                record=json.loads(result.stdout.strip().splitlines()[-1])
                record.update(input=str(source),input_sha256=sha(source),output=str(output),output_sha256=sha(transformed),
                    target=target,upstream_commit='ee05a48d79cbae11351e9c8eb711f286c1206402',
                    pass_binary_sha256=binary_identity,compiler_sources=source_identity,runtime='common/cpu-coarsening/runtime/cpu_runtime.cpp',
                    shared_memory_bytes=shared_memory_bytes,fallback='none',validation='LLVM verification only; numerical acceptance is workload-specific')
                if obj:
                    obj=Path(obj).resolve()
                    if obj==source or obj==output:raise ValueError('Object path must differ from IR paths')
                    staged=temp/'kernel.o'
                    command(['clang-18','--target='+target,'-Wno-override-module','-O3','-fPIC','-c',transformed,'-o',staged],300)
                    symbols=command(['llvm-nm-18','--undefined-only',staged],30).stdout
                    if re.search(r'pthread_|shfl_|warp_group_gather|launchKernel|region_sum',symbols):raise RuntimeError('Unlowered lane runtime in CPU object')
                    obj.parent.mkdir(parents=True,exist_ok=True);obj.write_bytes(staged.read_bytes())
                    record.update(object=str(obj),object_sha256=sha(obj),undefined_symbols=symbols.splitlines())
                output.write_bytes(transformed.read_bytes());manifest.write_text(json.dumps(record,indent=2)+'\n')
                return record
            except Exception as error:
                if transformed.exists(): output.with_suffix('.rejected.ll').write_bytes(transformed.read_bytes())
                # Never overwrite an earlier accepted artifact on a refused try.
                output.with_suffix('.refused.json').write_text(json.dumps({'status':'RECOGNISES-DOES-NOT-ADMIT',
                    'input':str(source),'kernel':kernel,'reason':str(error),'fallback':'none'},indent=2)+'\n')
                raise
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input');p.add_argument('--kernel',required=True)
    p.add_argument('-o','--output',required=True);p.add_argument('--object');p.add_argument('--rename');p.add_argument('--block-size',type=int,default=32)
    p.add_argument('--target',choices=['x86_64-linux-gnu','aarch64-linux-gnu'],default='x86_64-linux-gnu')
    p.add_argument('--shared-memory-bytes',type=int,default=32768)
    a=p.parse_args()
    try:r=compile_kernel(a.input,a.kernel,a.output,a.block_size,a.target,a.rename,a.object,a.shared_memory_bytes)
    except Exception as e:p.exit(2,'RECOGNISES-DOES-NOT-ADMIT: '+str(e)+'\n')
    print(json.dumps(r))
if __name__=='__main__':main()
