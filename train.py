"""Portable launcher for the RADVL reference model. No training occurs on import."""
import argparse,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/reference.json')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--output',type=Path,default=ROOT/'outputs')
    p.add_argument('--split-seed',type=int,default=1234)
    p.add_argument('--save-checkpoints',choices=('on','off'),default='off')
    a=p.parse_args()
    params=json.loads(a.config.read_text(encoding='utf-8'))
    params['data_path']=str(a.data_dir.resolve())
    required=['association.csv','gene_seq_sim.csv','drug_str_sim.csv']
    missing=[n for n in required if not (a.data_dir/n).is_file()]
    if missing:p.error('Missing input files: '+', '.join(missing)+'. See data/README.md.')
    protocol=ROOT/'protocols'/f"{params.get('protocol_name','edge_val125')}_seed{a.split_seed}"
    if not (protocol/'manifest.json').is_file():p.error('Prepare the input partition first with prepare_inputs.py; see README.md.')
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    cfg=output/f'parameters_split{a.split_seed}.json'
    cfg.write_text(json.dumps(params,indent=2),encoding='utf-8')
    command=[sys.executable,'-B','-m','radvl.train_entry','--parameters',str(cfg),'--output',str(output/f'split{a.split_seed}'),'--split_seed',str(a.split_seed),'--save_checkpoints',a.save_checkpoints]
    subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GRADIENT_AUDIT_EVERY='0'),check=True)

if __name__=='__main__':main()
