#!/usr/bin/env python3
"""Local caption events from an OBS clip or explicit stereo game-audio bus."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import queue
import re
import time
import uuid
import numpy as np
from websockets.asyncio.server import serve
from caption_policy import load_policy
from labeler import ROOT
from live_engine import LiveEngine
from model_tools import decode
from obs_source import ObsClient
from overlay_server import overlay_service
from label_replay import LabelTimeline, SpeechTimeline, ReplaySettings, SOUNDS, load_labels, load_speech


class SealedMediaError(ValueError):pass


def guard_media(path):
    candidate=Path(path).expanduser().resolve()
    manifest=json.loads((ROOT/'splits.json').read_text())
    sealed={(ROOT/p).resolve() for group in manifest['sealed_tests'].values() for p in group}
    if candidate in sealed:raise SealedMediaError('Sealed test media is blocked until final evaluation')
    # Resolve symlinks and reject hard-link aliases without reading any sealed audio.
    if not candidate.is_file():raise ValueError('Media file does not exist')
    stat=candidate.stat()
    for p in sealed:
        if p.exists() and (p.stat().st_dev,p.stat().st_ino)==(stat.st_dev,stat.st_ino):
            raise SealedMediaError('Sealed test media is blocked until final evaluation')
    return candidate


def load_env():
    path=ROOT/'.env'
    if path.exists():
        for row in path.read_text().splitlines():
            key,sep,value=row.partition('=')
            if sep and key.strip() in {'OBS_WS_URL','OBS_WS_PASSWORD'}:
                os.environ.setdefault(key.strip(),value.strip().strip('"').strip("'"))


class EventHub:
    def __init__(self,json_output=True):
        self.json_output=json_output
        self.clients={};self.active=[];self.speech=None;self.status={'type':'status','phase':'starting'}
        self.session=uuid.uuid4().hex;self.sequence=0

    def publish(self,event):
        event={**event,'session':self.session,'policy_version':2}
        if event['type']=='caption':
            self.active=([r for r in self.active if r['expires_at']>time.time()]+[event])[-2:]
        elif event['type']=='speech':self.speech=event if event.get('text') else None
        elif event['type']=='reset':self.active=[];self.speech=None
        elif event['type']=='status':self.status=event
        if self.json_output:print(json.dumps(event),flush=True)
        elif event['type']=='caption':print(f"#{event['id']} {event['text']}  ({event['source']} @ {event['start_s']:.3f}s)",flush=True)
        elif event['type']=='status':print(f"{event['phase']}: {event.get('source',event.get('websocket',''))}",flush=True)
        elif event['type']=='reset':print(f"Reset: {event['reason']}",flush=True)
        elif event['type']=='performance':print(f"Inference: median {event['processing_median_ms']:.1f}ms, p95 {event['processing_p95_ms']:.1f}ms; audio window 975ms",flush=True)
        for ws,q in list(self.clients.items()):
            try:q.put_nowait(event)
            except asyncio.QueueFull:
                self.clients.pop(ws,None)
                asyncio.create_task(ws.close(code=1013,reason='Slow consumer; reconnect for current snapshot'))

    async def handler(self,ws):
        q=asyncio.Queue(maxsize=32);self.clients[ws]=q
        try:
            await ws.send(json.dumps({'type':'snapshot','session':self.session,'policy_version':2,'status':self.status,
                                     'captions':[r for r in self.active if r['expires_at']>time.time()],
                                     'speech':self.speech if self.speech and self.speech['expires_at']>time.time() else None}))
            async def writer():
                while True:await asyncio.wait_for(ws.send(json.dumps(await q.get())),2)
            async def reader():
                async for _ in ws:pass  # Receive-only API: no remote commands, file reads or training.
            tasks=[asyncio.create_task(writer()),asyncio.create_task(reader())]
            try:
                done,_=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if not task.cancelled():task.result()
            finally:
                for task in tasks:task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
        except Exception:
            pass  # A disconnected/slow overlay must never stop inference.
        finally:self.clients.pop(ws,None)


class Runner:
    def __init__(self,engine,hub):self.engine=engine;self.hub=hub;self.source='';self.last_phase=None

    def status(self,phase,**extra):
        if phase!=self.last_phase:
            self.hub.publish({'type':'status','phase':phase,'source':self.source,**extra});self.last_phase=phase

    def reset(self,reason):
        self.engine.reset();self.hub.publish({'type':'reset','reason':reason,'source':self.source})

    async def frame(self,stereo,end_s,source_lag_ms=0):
        captions,elapsed=await asyncio.to_thread(self.engine.window,stereo,end_s)
        for row in captions:
            self.hub.sequence+=1
            side={'left':' ← left','right':' → right'}.get(row['direction'],'')
            now=time.time()
            self.hub.publish({'type':'caption','id':self.hub.sequence,'source':self.source,**row,
                'text':f"[{row['label']}{side}]",'emitted_at':now,'expires_at':now+self.engine.deployment['display_duration_s'],
                'processing_ms':round(elapsed,3),'source_lag_ms':round(source_lag_ms+elapsed,3)})


async def file_source(runner,path):
    path=guard_media(path);runner.source=path.name;runner.status('loading')
    audio=await asyncio.to_thread(decode,path,runner.engine.rate)
    if not len(audio):raise ValueError('Media has no audio')
    runner.reset('File playback started');runner.status('listening')
    clock=time.monotonic();last_end=-1
    for offset in range(0,len(audio),runner.engine.hop):
        end=min(offset+runner.engine.size,len(audio));end_s=end/runner.engine.rate
        if end_s<=last_end:break
        await asyncio.sleep(max(0,clock+end_s-time.monotonic()))
        lag=max(0,(time.monotonic()-clock-end_s)*1000)
        if lag>runner.engine.hop/runner.engine.rate*1000:
            runner.reset('Inference fell behind; skipping stale file audio')
            continue
        await runner.frame(audio[offset:end],end_s,lag);last_end=end_s


async def device_source(runner,index):
    import sounddevice as sd
    from scipy.signal import resample_poly
    device=sd.query_devices(index,'input')
    if device['max_input_channels']<2:raise ValueError('Choose a stereo game-only input; mono microphones are rejected')
    rate=int(device['default_samplerate']);block=int(rate/50)
    if rate%50 or not 8000<=rate<=192000:raise ValueError('Choose a game bus at 16, 44.1, 48 or 96 kHz')
    sd.check_input_settings(device=index,channels=2,dtype='float32',samplerate=rate)
    chunks=queue.Queue(maxsize=50);native_position=0
    def callback(indata,frames,timing,status):
        nonlocal native_position
        position=native_position;native_position+=frames
        try:chunks.put_nowait((position,indata.copy(),bool(status),time.monotonic()))
        except queue.Full:pass  # A missing sample index triggers an explicit reset below.
    runner.source=device['name'];runner.status('listening');runner.reset('Stereo game-audio capture started')
    buffer=np.empty((0,2),np.float32);origin=0;expected=None;gcd=math.gcd(rate,runner.engine.rate)
    native_size=round(runner.engine.size*rate/runner.engine.rate)
    native_hop=round(runner.engine.hop*rate/runner.engine.rate)
    with sd.InputStream(device=index,channels=2,samplerate=rate,dtype='float32',blocksize=block,latency='low',callback=callback):
        while True:
            try:
                # Drop stale capture blocks instead of letting caption lag grow.
                if chunks.qsize()>10:
                    while chunks.qsize()>1:chunks.get_nowait()
                position,stereo,overflow,received_at=chunks.get_nowait()
            except queue.Empty:
                await asyncio.sleep(.005);continue
            if overflow or (expected is not None and position!=expected):
                runner.reset('Audio input gap or overflow');buffer=np.empty((0,2),np.float32)
            if not len(buffer):origin=position/rate
            expected=position+len(stereo)
            buffer=np.concatenate([buffer,stereo])
            while len(buffer)>=native_size:
                frame=buffer[:native_size]
                # Resample whole overlapping model windows, not isolated 20ms
                # chunks: repeated chunk-edge zero padding would add fake transients.
                if rate!=runner.engine.rate:
                    frame=await asyncio.to_thread(resample_poly,frame,runner.engine.rate//gcd,rate//gcd,axis=0)
                    frame=frame[:runner.engine.size]
                await runner.frame(frame,origin+native_size/rate,max(0,(time.monotonic()-received_at)*1000))
                buffer=buffer[native_hop:];origin+=native_hop/rate


async def obs_source(runner,name):
    backoff=.5
    while True:
        try:
            async with ObsClient() as obs:
                path=None;audio=None;next_offset=None;last_cursor=None;last_poll=None;state=None
                runner.source=name;runner.reset('OBS connected');runner.status('connected')
                while True:
                    # Read-only: never start/stop a stream, alter a scene or control playback.
                    settings=await obs.request('GetInputSettings',{'inputName':name})
                    local=settings.get('inputSettings',{}).get('local_file')
                    if settings.get('inputKind')!='ffmpeg_source' or not local:
                        raise ValueError('Select a local OBS Media Source with a video/audio file')
                    checked=guard_media(local)
                    if checked!=path:
                        path=checked;runner.source=path.name;runner.status('loading')
                        audio=await asyncio.to_thread(decode,path,runner.engine.rate)
                        if not len(audio):raise ValueError('Selected OBS source has no audio')
                        next_offset=None;last_cursor=None;runner.reset('OBS media file changed')
                    media=await obs.request('GetMediaInputStatus',{'inputName':name})
                    current_state=media['mediaState'];cursor=media.get('mediaCursor');poll=time.monotonic()
                    if current_state!='OBS_MEDIA_STATE_PLAYING' or cursor is None:
                        if state!=current_state:runner.reset('OBS playback paused/stopped')
                        runner.status('paused' if current_state=='OBS_MEDIA_STATE_PAUSED' else 'waiting-for-playback')
                        next_offset=None;last_cursor=None;state=current_state
                        await asyncio.sleep(.1);continue
                    cursor=cursor/1000
                    discontinuity=(last_cursor is not None and (cursor<last_cursor-.1 or cursor-last_cursor>poll-last_poll+.35))
                    if discontinuity:
                        runner.reset('OBS seek or loop');next_offset=None
                    if next_offset is not None and cursor-(next_offset+runner.engine.size)/runner.engine.rate>.75:
                        runner.reset('Skipping stale OBS audio');next_offset=None
                    if next_offset is None:
                        next_offset=max(0,int((cursor-runner.engine.size/runner.engine.rate)/(runner.engine.hop/runner.engine.rate)))*runner.engine.hop
                    runner.status('listening');state=current_state;last_cursor=cursor;last_poll=poll;backoff=.5
                    end=next_offset+runner.engine.size
                    if end<=len(audio) and end/runner.engine.rate<=cursor:
                        await runner.frame(audio[next_offset:end],end/runner.engine.rate,max(0,(cursor-end/runner.engine.rate)*1000))
                        next_offset+=runner.engine.hop
                    await asyncio.sleep(.04)
        except (OSError,TimeoutError) as exc:
            runner.reset('OBS connection lost');runner.status('reconnecting',message='Start OBS and check local WebSocket settings')
            await asyncio.sleep(backoff);backoff=min(5,backoff*2)
        except Exception as exc:
            from websockets.exceptions import ConnectionClosed
            if not isinstance(exc,ConnectionClosed):raise
            runner.reset('OBS connection closed');runner.status('reconnecting',message='Check OBS WebSocket password/settings')
            await asyncio.sleep(backoff);backoff=min(5,backoff*2)


async def current_scene_media(obs):
    scene = (await obs.request('GetCurrentProgramScene'))['currentProgramSceneName']
    items = (await obs.request('GetSceneItemList', {'sceneName': scene}))['sceneItems']
    media = [item['sourceName'] for item in items
             if item.get('sceneItemEnabled') and item.get('inputKind') == 'ffmpeg_source']
    return scene, media[0] if len(media) == 1 else None


async def replay_source(runner, name):
    """Follow the actual OBS playhead using human annotations, not predictions."""
    from websockets.exceptions import ConnectionClosed
    backoff = .5
    while True:
        try:
            async with ObsClient() as obs:
                path = timeline = speech = selection = None
                runner.reset('Hand-label replay connected (no model inference)')
                while True:
                    selected = (None, name) if name else await current_scene_media(obs)
                    if selected != selection:
                        selection = selected
                        path = timeline = speech = None
                        runner.last_phase = None
                        runner.reset('Active OBS scene changed' if not name else 'Replay source selected')
                    active_name = selected[1]
                    if active_name is None:
                        runner.source = selected[0]
                        runner.status('waiting-for-scene', mode='csv_replay',
                                      message='Active scene needs one visible local Media Source')
                        await asyncio.sleep(.1)
                        continue
                    settings = await obs.request('GetInputSettings', {'inputName': active_name})
                    local = settings.get('inputSettings', {}).get('local_file')
                    if settings.get('inputKind') != 'ffmpeg_source' or not local:
                        raise ValueError('Replay requires a local OBS Media Source')
                    checked = guard_media(local)  # Before opening any CSV, including after a source change.
                    if checked != path:
                        rows = load_labels(checked)
                        path = checked
                        runner.source = path.name
                        runner.last_phase = None
                        timeline = LabelTimeline(rows, runner.engine.deployment['display_duration_s'])
                        speech = SpeechTimeline(load_speech(checked))
                        runner.reset('Hand-label replay file changed')
                    media = await obs.request('GetMediaInputStatus', {'inputName': active_name})
                    playing = media['mediaState'] == 'OBS_MEDIA_STATE_PLAYING'
                    cursor = media.get('mediaCursor')
                    cursor = cursor / 1000 if cursor is not None else None
                    update = timeline.update(cursor, playing, time.monotonic())
                    if update.reason:
                        runner.reset(update.reason)
                    runner.status('replaying' if playing else 'paused', mode='csv_replay')
                    for row in update.captions:
                        row = dict(row)
                        remaining = row.pop('remaining_s')
                        side = {'left': ' ← left', 'right': ' → right'}.get(row['direction'], '')
                        now = time.time()
                        runner.hub.sequence += 1
                        runner.hub.publish({'type': 'caption', 'id': runner.hub.sequence,
                            'mode': 'csv_replay', 'source': runner.source, **row,
                            'text': f"[{row['label']}{side}]", 'emitted_at': now,
                            'expires_at': now + remaining, 'processing_ms': 0,
                            'source_lag_ms': round(max(0, cursor - row['start_s']) * 1000, 3)})
                    changed, cue = speech.update(cursor, playing, force=bool(update.reason),
                        speed=float(settings.get('inputSettings', {}).get('speed_percent', 100)) / 100)
                    if changed:
                        now = time.time()
                        if cue:
                            cue = dict(cue)
                            remaining = cue.pop('remaining_s')
                            runner.hub.publish({'type': 'speech', 'mode': 'csv_replay',
                                'source': runner.source, **cue,
                                'emitted_at': now, 'expires_at': now + remaining})
                        else:
                            runner.hub.publish({'type': 'speech', 'mode': 'csv_replay',
                                'source': runner.source, 'text': '',
                                'emitted_at': now, 'expires_at': now})
                    backoff = .5
                    await asyncio.sleep(.04)
        except (OSError, TimeoutError, ConnectionClosed):
            runner.reset('Hand-label replay connection lost')
            runner.status('reconnecting', mode='csv_replay')
            await asyncio.sleep(backoff)
            backoff = min(5, backoff * 2)


async def run(args):
    load_env()
    if args.list_obs_inputs:
        async with ObsClient() as obs:
            for source in (await obs.request('GetInputList'))['inputs']:
                if source['inputKind']=='ffmpeg_source':print(source['inputName'])
        return
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices());return
    engine = ReplaySettings() if args.replay_labels else await asyncio.to_thread(LiveEngine,args.show_centered)
    if args.replay_labels and not args.json:
        print('HAND-LABEL REPLAY: CSV annotations, including centered sounds; no model inference.', flush=True)
    hub=EventHub(json_output=args.json);runner=Runner(engine,hub)
    origins=[None,'null',re.compile(r'http://(127\.0\.0\.1|localhost)(:[0-9]+)?')]
    with overlay_service(args.overlay_port,args.port):
        await serve_source(args,engine,hub,runner,origins)


async def serve_source(args,engine,hub,runner,origins):
    async with serve(hub.handler,'127.0.0.1',args.port,origins=origins,max_size=4096,max_queue=4,close_timeout=1):
        hub.publish({'type':'status','phase':'ready','websocket':f'ws://127.0.0.1:{args.port}',
            'mode':'csv_replay' if args.replay_labels else 'inference',
            'model':None if args.replay_labels else f"round-{engine.deployment['model_round']}",'thresholds':engine.deployment['thresholds'],
            'overlay':f'http://127.0.0.1:{args.overlay_port}/overlay','display_duration_s':engine.deployment['display_duration_s'],
            'enabled_labels':sorted(SOUNDS) if args.replay_labels else load_policy()['enabled_labels'],
            'show_centered':args.replay_labels or args.show_centered})
        source = (replay_source(runner, None if getattr(args, 'follow_scene', False) else args.obs_media) if args.replay_labels else
                  file_source(runner,args.file) if args.file else
                  obs_source(runner,args.obs_media) if args.obs_media else device_source(runner,args.device))
        try:
            if args.seconds:await asyncio.wait_for(source,args.seconds)
            else:await source
        except asyncio.TimeoutError:pass
        finally:
            runner.reset('Helper stopped');runner.status('stopped')
            if engine.compute_ms:
                hub.publish({'type':'performance','windows':len(engine.compute_ms),
                    'processing_median_ms':round(float(np.median(engine.compute_ms)),3),
                    'processing_p95_ms':round(float(np.percentile(engine.compute_ms,95)),3),
                    'audio_window_ms':975,'hop_ms':480})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--file',help='Paced local clip; does not play sound or control OBS')
    mode.add_argument('--obs-media',help='Exact name of the local OBS Media Source to follow')
    mode.add_argument('--follow-scene', action='store_true',
                      help='Follow the visible Media Source in the live OBS scene (with --replay-labels)')
    mode.add_argument('--device',type=int,help='Explicit stereo game-only audio input ID; never defaults to microphone')
    mode.add_argument('--list-devices',action='store_true')
    mode.add_argument('--list-obs-inputs',action='store_true')
    parser.add_argument('--port',type=int,default=8770)
    parser.add_argument('--overlay-port',type=int,default=8771)
    parser.add_argument('--json',action='store_true',help='Machine-readable console events instead of plain captions')
    parser.add_argument('--show-centered',action='store_true',help='Include centered sounds without an arrow; hidden by default')
    parser.add_argument('--replay-labels', action='store_true',
                        help='With --obs-media: replay the adjacent hand-labeled CSV, without model inference')
    parser.add_argument('--seconds',type=float,help='Stop after this many seconds (smoke testing)')
    args=parser.parse_args()
    if args.replay_labels and not (args.obs_media or args.follow_scene):
        parser.error('--replay-labels requires --obs-media or --follow-scene')
    if args.follow_scene and not args.replay_labels:
        parser.error('--follow-scene requires --replay-labels')
    if args.seconds is not None and args.seconds<=0:parser.error('--seconds must be positive')
    try:asyncio.run(run(args))
    except KeyboardInterrupt:pass
    except Exception as exc:parser.exit(1,f'Helper stopped: {exc}\n')


if __name__=='__main__':main()
