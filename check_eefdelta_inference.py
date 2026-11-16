"""Real weights and recorded images only. No robot SDK/Engine/Camera imports."""
import gc
import importlib.util
import json
from pathlib import Path
import time

from pi05_policy import Pi05Policy, ROOT
from check_tcp_inference import samples
from model_contract import (EEFDELTA, TCP, tcp_state_to_eefdelta,
                            decode_eefdelta_actions, tcp_action_to_sdk)
from inference_settings import InferenceSettings
import numpy as np
import torch


def main():
    snapshot = ROOT.parent / '模型下载_EEFDelta_20261022/training_snapshot'
    spec = importlib.util.spec_from_file_location('training_pose_contract', snapshot/'pose_contract.py')
    ref = importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    models = sorted((snapshot.parent).glob('pi05*/*.pt'))
    # Also exercise old modes through the same updated loader/policy.
    models += sorted((ROOT.parent/'模型下载_TCP_20261021').glob('*best*/*.pt'))
    models += sorted((ROOT.parent/'模型下载_20261020').glob('*best*/*.pt'))
    recorded = samples()
    report = dict(hardware_connected=False, cuda_device=torch.cuda.get_device_name(0),
                  purpose='strict checkpoint loading and recorded-frame inference, not real robot success',models=[])
    output = ROOT/'eefdelta_inference_check.json'
    options = InferenceSettings(denoise_steps=10,execute_frames=5).to_dict()
    for filename in models:
        print('MODEL', filename.name,flush=True)
        started=time.monotonic()
        policy=Pi05Policy(filename,lambda msg:print(msg,flush=True))
        mode=policy.contract['mode']
        item=dict(model=str(filename),mode=mode,load_seconds=time.monotonic()-started,
                  sha256=policy.meta['sha256'],state_dim=len(policy.stats['state']['mean']),samples=[])
        for frame,images,tcp,joints in recorded:
            state=tcp_state_to_eefdelta(tcp) if mode==EEFDELTA else tcp if mode==TCP else joints
            if mode==EEFDELTA:np.testing.assert_allclose(state,ref.encode_state(tcp),atol=1e-7)
            actions,seconds=policy.predict(*images,state,(ROOT/'assets/task.txt').read_text(),options)
            assert actions.shape==(50,7 if mode==EEFDELTA else 8) and np.isfinite(actions).all()
            entry=dict(frame=frame,state_shape=list(state.shape),model_shape=list(actions.shape),
                       seconds=seconds,finite=True,first_model_action=actions[0].tolist())
            if mode==EEFDELTA:
                decoded=decode_eefdelta_actions(state,actions)
                expected=ref.decode_actions(state,actions)
                np.testing.assert_allclose(decoded[:,:3],expected[:,:3],atol=2e-7)
                np.testing.assert_allclose(ref.quat_matrix(decoded[:,3:7]),ref.quat_matrix(expected[:,3:7]),atol=3e-7)
                np.testing.assert_allclose(decoded[:,7],actions[:,6],atol=1e-7)
                entry.update(decoded_shape=list(decoded.shape),reference_match=True,
                             first_absolute_action=decoded[0].tolist())
                for row in decoded:assert np.isfinite(tcp_action_to_sdk(row)).all()
            item['samples'].append(entry)
            print('frame',frame,'shape',actions.shape,'seconds',round(seconds,3),flush=True)
        report['models'].append(item)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        del policy
        gc.collect();torch.cuda.empty_cache()
    assert len(report['models'])==5
    report['complete']=True
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('PASS',output,flush=True)


if __name__=='__main__':main()
