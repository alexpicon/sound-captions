import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from websockets.asyncio.server import serve
from websockets.asyncio.client import connect
from caption_policy import apply_policy
from live_engine import WindowScheduler
from live_helper import EventHub,guard_media,SealedMediaError
from obs_source import ObsClient,auth_response
from run_baseline import captions_from_windows


class StreamingTests(unittest.TestCase):
    def test_incremental_scheduler_and_filter_equal_batch_on_random_windows(self):
        config={'class_mapping':{'footsteps':[0],'gunshot':[1],'reload':[2],'explosion':[3]},
                'thresholds':dict.fromkeys(['footsteps','gunshot','reload','explosion'],.9),
                'caption_ttl_s':2,'max_captions':2,'hide_centered':True}
        rng=np.random.default_rng(42)
        windows=[{'time':i*.48+.975,'direction':rng.choice(['left','right','center','unknown']),
                  'scores':rng.random(4)} for i in range(300)]
        batch=captions_from_windows(windows,config);engine=WindowScheduler(config)
        stream=[r for w in windows for r in engine.step(w['time'],w['direction'],w['scores'])]
        self.assertEqual(stream,batch);self.assertEqual(apply_policy(stream),apply_policy(batch))

    def test_sealed_paths_symlinks_and_hardlinks_rejected_before_decode(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sealed=root/'sealed.wav';sealed.write_bytes(b'fake fixture')
            (root/'splits.json').write_text(json.dumps({'sealed_tests':{'real':['sealed.wav']}}))
            symbolic=root/'alias.wav';symbolic.symlink_to(sealed)
            hard=root/'hard.wav';os.link(sealed,hard)
            with patch('live_helper.ROOT',root):
                for path in [sealed,symbolic,hard]:
                    with self.assertRaises(SealedMediaError):guard_media(path)
                allowed=root/'train.wav';allowed.write_bytes(b'fake train fixture')
                self.assertEqual(guard_media(allowed),allowed.resolve())


class NetworkTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_event_delivery_and_reconnect_snapshot(self):
        hub=EventHub()
        async with serve(hub.handler,'127.0.0.1',0) as server:
            port=server.sockets[0].getsockname()[1]
            async with connect(f'ws://127.0.0.1:{port}',proxy=None) as client:
                self.assertEqual(json.loads(await client.recv())['type'],'snapshot')
                import time
                hub.publish({'type':'caption','id':1,'label':'gunshot','expires_at':time.time()+2})
                self.assertEqual(json.loads(await client.recv())['id'],1)
            async with connect(f'ws://127.0.0.1:{port}',proxy=None) as client:
                snapshot=json.loads(await client.recv());self.assertEqual(snapshot['captions'][0]['id'],1)
                hub.publish({'type':'reset','reason':'seek'})
                self.assertEqual(json.loads(await client.recv())['type'],'reset')
            async with connect(f'ws://127.0.0.1:{port}',proxy=None) as client:
                self.assertEqual(json.loads(await client.recv())['captions'],[])

    async def test_obs_v5_auth_and_read_only_requests_over_real_websocket(self):
        async def fake_obs(ws):
            await ws.send(json.dumps({'op':0,'d':{'rpcVersion':1,'authentication':{'salt':'salt','challenge':'challenge'}}}))
            identify=json.loads(await ws.recv())
            self.assertEqual(identify['d']['authentication'],auth_response('unit-password','salt','challenge'))
            self.assertEqual(identify['d']['eventSubscriptions'],0)
            await ws.send(json.dumps({'op':2,'d':{'negotiatedRpcVersion':1}}))
            async for raw in ws:
                req=json.loads(raw)['d'];self.assertEqual(req['requestType'],'GetMediaInputStatus')
                await ws.send(json.dumps({'op':7,'d':{'requestId':req['requestId'],'requestStatus':{'result':True,'code':100},'responseData':{'mediaState':'OBS_MEDIA_STATE_PAUSED','mediaCursor':2000}}}))
        async with serve(fake_obs,'127.0.0.1',0) as server:
            port=server.sockets[0].getsockname()[1]
            async with ObsClient(f'ws://127.0.0.1:{port}','unit-password') as obs:
                result=await obs.request('GetMediaInputStatus',{'inputName':'Clip'})
                self.assertEqual(result['mediaCursor'],2000)
                with self.assertRaises(ValueError):await obs.request('StartStream')

    async def test_obs_media_pause_seek_and_reconnect_resets(self):
        from live_helper import obs_source,Runner
        resets=[];frames=[];statuses=[]
        class Engine:
            rate=16000;size=15600;hop=7680
            def reset(self):resets.append('reset')
            def window(self,stereo,t):frames.append(t);return [],1
        class Hub:
            def publish(self,event):statuses.append(event)
        class FakeObs:
            index=0;connections=0
            async def __aenter__(self):FakeObs.connections+=1;return self
            async def __aexit__(self,*args):pass
            async def request(self,kind,data):
                if kind=='GetInputSettings':return {'inputKind':'ffmpeg_source','inputSettings':{'local_file':'fake.wav'}}
                values=[('OBS_MEDIA_STATE_PLAYING',1000),('OBS_MEDIA_STATE_PLAYING',1500),('OBS_MEDIA_STATE_PLAYING',100),('OBS_MEDIA_STATE_PAUSED',100)]
                i=FakeObs.index;FakeObs.index+=1
                if i>=len(values):raise OSError('mock disconnect')
                state,cursor=values[i];return {'mediaState':state,'mediaCursor':cursor}
        runner=Runner(Engine(),Hub())
        with patch('live_helper.ObsClient',FakeObs),patch('live_helper.guard_media',return_value=Path('fake.wav')),patch('live_helper.decode',return_value=np.zeros((48000,2),np.float32)):
            with self.assertRaises(asyncio.TimeoutError):await asyncio.wait_for(obs_source(runner,'Clip'),.9)
        self.assertEqual(frames,[.975,1.455])
        reasons=[e['reason'] for e in statuses if e['type']=='reset']
        self.assertIn('OBS seek or loop',reasons);self.assertIn('OBS playback paused/stopped',reasons);self.assertIn('OBS connection lost',reasons)
        self.assertGreaterEqual(FakeObs.connections,2)


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_stereo_native_windows_resample_without_chunk_boundary_transients(self):
        from live_helper import device_source
        captured=[];opened=[]
        class Runner:
            engine=type('Engine',(),{'rate':16000,'size':15600,'hop':7680})()
            def status(self,*args,**kwargs):pass
            def reset(self,*args):pass
            async def frame(self,data,end,lag):captured.append(data.copy())
        class Stream:
            def __init__(self,**kwargs):self.callback=kwargs['callback'];opened.append(kwargs)
            def __enter__(self):
                async def feed():
                    for i in range(50):
                        data=np.full((960,2),.6,np.float32);data[:,1]=.2
                        self.callback(data,960,None,False)
                        await asyncio.sleep(.003)
                self.task=asyncio.create_task(feed());return self
            def __exit__(self,*args):self.task.cancel()
        with patch('sounddevice.query_devices',return_value={'name':'fake game bus','max_input_channels':2,'default_samplerate':48000}),patch('sounddevice.check_input_settings'),patch('sounddevice.InputStream',Stream):
            with self.assertRaises(asyncio.TimeoutError):await asyncio.wait_for(device_source(Runner(),77),.6)
        self.assertEqual(opened[0]['device'],77);self.assertEqual(opened[0]['channels'],2)
        self.assertEqual(len(captured),1);self.assertEqual(captured[0].shape,(15600,2))
        self.assertAlmostEqual(float(np.median(captured[0][:,0])),.6,places=3)
        self.assertAlmostEqual(float(np.median(captured[0][:,1])),.2,places=3)
        self.assertLess(float(np.std(captured[0][100:-100,0])),1e-5)

    async def test_mono_input_rejected_before_opening_capture(self):
        from live_helper import device_source
        with patch('sounddevice.query_devices',return_value={'max_input_channels':1}),patch('sounddevice.InputStream') as stream:
            with self.assertRaises(ValueError):await device_source(None,77)
            stream.assert_not_called()
