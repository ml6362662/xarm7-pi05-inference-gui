"""Fixed-anchor/frame counterexamples and complete mocked execution, no hardware."""
import hashlib
import importlib.util
import json
from pathlib import Path
import queue
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from model_contract import (EEFDELTA, LABELS, DELTA_STATE, DELTA_ACTION, DELTA_ROTATION,
    QUATERNION_POLICY, tcp_state_to_eefdelta, eefdelta_state_to_tcp, decode_eefdelta_actions,
    orientation_gap_deg, tcp_action_to_sdk)
from pi05_policy import read_bundle, normalize_state, prompt_tokens
from test_control import bare_engine
from test_tcp_adapter import tcp_robot, FRAME

ROOT = Path(__file__).parent
CONTRACT = dict(mode=EEFDELTA, label=LABELS[EEFDELTA], robot_frame=FRAME)


def reference():
    path = ROOT.parent / '模型下载_EEFDelta_20261022/training_snapshot/pose_contract.py'
    spec = importlib.util.spec_from_file_location('training_pose_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matches_archived_contract_for_noncommuting_rotations_and_fixed_anchor():
    ref = reference()
    rng = np.random.default_rng(42)
    for i in range(12):
        q = Rotation.random(random_state=rng).as_quat()[[3, 0, 1, 2]]
        raw = np.r_[rng.uniform(-.3, .6, 3), q, .35]
        state = tcp_state_to_eefdelta(raw)
        np.testing.assert_allclose(state, ref.encode_state(raw), atol=1e-7)
        a = np.c_[rng.normal(0, .05, (50, 3)), rng.normal(0, .4, (50, 3)), rng.uniform(0, 1, 50)]
        decoded = decode_eefdelta_actions(state, a)
        expected = ref.decode_actions(state, a)
        np.testing.assert_allclose(decoded[:, :3], expected[:, :3], atol=1e-7)
        np.testing.assert_allclose(ref.quat_matrix(decoded[:, 3:7]), ref.quat_matrix(expected[:, 3:7]), atol=2e-7)
        np.testing.assert_allclose(decoded[:, 7], a[:, 6], atol=1e-7)
        # The second target is relative to measured anchor, not the first target.
        np.testing.assert_allclose(decoded[1, :3], state[:3] + a[1, :3], atol=1e-7)
        assert not np.allclose(decoded[1, :3], decoded[0, :3] + a[1, :3])


def test_rot6d_columns_and_rotation_left_multiplication_not_local_frame():
    anchor = Rotation.from_euler('xyz', [.6, 1.1, -.4])
    raw = np.r_[.2, -.1, .4, anchor.as_quat()[[3, 0, 1, 2]], .2]
    s = tcp_state_to_eefdelta(raw)
    np.testing.assert_allclose(s[3:6], anchor.as_matrix()[:, 0], atol=1e-7)
    np.testing.assert_allclose(s[6:9], anchor.as_matrix()[:, 1], atol=1e-7)
    dr = Rotation.from_rotvec([.3, -.2, .1])
    a = np.array([[.01, -.02, 0, .3, -.2, .1, .9]])
    decoded = decode_eefdelta_actions(s, a)[0]
    got = Rotation.from_quat(decoded[[4, 5, 6, 3]])
    assert (got.inv() * (dr * anchor)).magnitude() < 1e-6
    assert (got.inv() * (anchor * dr)).magnitude() > .1
    assert decoded[7] == pytest.approx(.9)  # Absolute grip, not anchor .2 plus .9.


@pytest.mark.parametrize('angles', [[0,0,0], [np.pi,0,0], [0,np.pi/2,.7]])
def test_zero_delta_and_quaternion_sign_equivalence(angles):
    q = Rotation.from_euler('xyz', angles).as_quat()[[3,0,1,2]]
    raw = np.r_[.1,.2,.3,q,.8]
    other = raw.copy();other[3:7] *= -1
    s = tcp_state_to_eefdelta(raw)
    np.testing.assert_array_equal(s, tcp_state_to_eefdelta(other))
    actions = np.tile([0,0,0,0,0,0,.4],(50,1))
    out = decode_eefdelta_actions(s, actions)
    np.testing.assert_allclose(out[:,:3], np.tile(raw[:3],(50,1)), atol=1e-7)
    assert orientation_gap_deg(raw[3:7],out[0,3:7]) < 1e-5


@pytest.mark.parametrize('bad', ['state_size','state_nan','zero_column','parallel_columns','action_size','action_inf'])
def test_invalid_input_rejected_before_queuing(bad):
    e = bare_engine();e.model_contract=CONTRACT
    s = tcp_state_to_eefdelta([.2,.3,.4,1,0,0,0,1])
    a = np.zeros((50,7))
    if bad=='state_size':s=s[:8]
    if bad=='state_nan':s[0]=np.nan
    if bad=='zero_column':s[3:6]=0
    if bad=='parallel_columns':s[6:9]=s[3:6]
    if bad=='action_size':a=np.zeros((50,8))
    if bad=='action_inf':a[0,0]=np.inf
    e.pending.update(state=s,joint_state=np.zeros(7))
    with pytest.raises((ValueError,RuntimeError)):
        e.result(dict(kind='actions',token=2,actions=a,elapsed=.1))
    assert not e.actions


def test_observation_decode_execution_and_log_keep_same_anchor():
    e=bare_engine();e.model_contract=CONTRACT
    e.robot,calls,writes=tcp_robot();e.robot.configure_policy(CONTRACT)
    image=np.zeros((4,4,3),np.uint8);stamp=time.monotonic()
    e.cameras={k:SimpleNamespace(snapshot=lambda:(image,stamp)) for k in ('global','wrist')}
    e.pending=e.worker=None;e.next_action=0;e.requests=queue.Queue();e.prompt='test'
    logs=[];e.record=lambda kind,data,*args,**kwargs:logs.append((kind,data))
    # Persisted UI settings may be higher than this checkpoint's five-frame
    # recommendation; the engine must enforce the contract when queuing.
    e.settings=replace(e.settings,execute_frames=10)
    e.tick();req=e.requests.get_nowait()
    assert req['state'].shape==(10,)
    np.testing.assert_allclose(req['state'],[0,0,.5,1,0,0,0,1,0,1])
    a=np.tile([.001,0,0,0,0,0,1.],(50,1));a[1,0]=.002
    e.result(dict(kind='actions',token=e.generation,actions=a,elapsed=.1))
    assert len(e.actions)==5
    assert e.inference_info['execute_frames']==5
    np.testing.assert_allclose(e.actions[1][:3],[.002,0,.5],atol=1e-7)
    pred=next(data for kind,data in logs if kind=='prediction')
    assert np.shape(pred['model_actions'])==(50,7)
    assert np.shape(pred['actions'])==(50,8)
    assert pred['anchor_state']==req['state'].tolist()
    assert e.inference_info['shape']==(50,7)
    assert e.inference_info['model_mode']==EEFDELTA
    e.tick()
    assert len(calls)==len(writes)==1
    np.testing.assert_allclose(calls[0][0][:3],[1,0,500],atol=1e-5)
    record=next(data for kind,data in logs if kind=='action')
    assert len(record['model_state'])==10 and record['model_mode']==EEFDELTA


def test_cancelled_delta_result_never_decodes_or_refills_queue():
    e=bare_engine();e.model_contract=CONTRACT
    e.request('pause')
    e.result(dict(kind='actions',token=2,actions=np.full((50,7),np.nan),elapsed=.1))
    assert not e.actions


def test_bundle_accepts_zero_padding_but_rejects_wrong_norm_and_rotation(tmp_path):
    path=tmp_path/'test.pt';path.write_bytes(b'test')
    config={'actor':{'model':{'action_dim':7,'num_action_chunks':50,'openpi':{
        'config_name':'pi05_xarm7_eefdelta_state','discrete_state_input':True,
        'num_images_in_input':2,'action_env_dim':7,'action_chunk':50,'max_token_len':200},
        'openpi_data':{'pad_state':False}}}}
    (tmp_path/'training_config.yaml').write_text(yaml.safe_dump(config))
    stats={k:{n:[v]*d for n,v in [('mean',0),('std',1),('q01',-1),('q99',1)]} for k,d in [('state',10),('actions',32)]}
    stats['actions']['q01'][7:]=stats['actions']['q99'][7:]=[0]*25
    stats_path=tmp_path/'norm_stats.json';stats_path.write_text(json.dumps({'norm_stats':stats}))
    (tmp_path/'download_verified.json').write_text(json.dumps({'bytes':4,'sha256':hashlib.sha256(b'test').hexdigest()}))
    spec=dict(state_semantics=DELTA_STATE,action_semantics=DELTA_ACTION,rotation_convention=DELTA_ROTATION,
        quaternion_policy=QUATERNION_POLICY,robot_frame=FRAME,state_dim=10,action_dim=7,model_action_dim=32,horizon=50)
    spec_path=tmp_path/'dataset_spec.json';spec_path.write_text(json.dumps(spec))
    _,norm,meta=read_bundle(path)
    assert meta['model_contract']['mode']==EEFDELTA
    assert normalize_state(np.zeros(10),norm['state']).shape==(10,)
    with pytest.raises(ValueError):normalize_state(np.zeros(8),norm['state'])
    spec['rotation_convention']='R_anchor @ Exp(rotvec)'
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='左乘'):read_bundle(path)
    spec['rotation_convention']=DELTA_ROTATION;spec_path.write_text(json.dumps(spec))
    stats['actions']['q99'][6]=-1;stats_path.write_text(json.dumps({'norm_stats':stats}))
    with pytest.raises(ValueError,match='归一化区间'):read_bundle(path)


def test_state_tokenization_preserves_all_ten_dimensions():
    seen=[]
    tokenizer=SimpleNamespace(encode=lambda text,**kw:seen.append(text) or [1,2])
    prompt_tokens(tokenizer,'test',np.linspace(-.9,.9,10))
    assert len(seen[0].split('State: ')[1].split(';')[0].split())==10
