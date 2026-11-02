import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.request
from deployment import load_deployment
from live_engine import WindowScheduler
from caption_policy import apply_policy
from overlay_server import make_server


class DeploymentTests(unittest.TestCase):
    def test_per_class_thresholds_and_checkpoint_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'model').write_bytes(b'model')
            cfg={'schema_version':1,'model':'model','model_sha256':hashlib.sha256(b'model').hexdigest(),
                 'thresholds':{'footsteps':.5,'gunshot':.9,'reload':.9,'explosion':.9},
                 'display_duration_s':3,'max_display_captions':2,'raw_scheduler_ttl_s':2,'frozen_sources':{}}
            file=root/'live-config.json';file.write_text(json.dumps(cfg))
            self.assertEqual(load_deployment(root=root)['thresholds']['footsteps'],.5)
            (root/'model').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):load_deployment(root=root)
            cfg['model']='../escape';file.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'local'):load_deployment(root=root)

    def test_lowering_only_footsteps_does_not_lower_gunshot_threshold(self):
        import numpy as np
        c={'class_mapping':{'footsteps':[0],'gunshot':[1],'reload':[2],'explosion':[3]},
           'thresholds':{'footsteps':.5,'gunshot':.9,'reload':.9,'explosion':.9},
           'hide_centered':True,'caption_ttl_s':2,'max_captions':2}
        rows=apply_policy(WindowScheduler(c).step(.975,'left',np.array([.6,.8,.99,0])))
        self.assertEqual([r['label'] for r in rows],['footsteps'])

    def test_hub_keeps_only_newest_two_for_reconnect(self):
        from live_helper import EventHub
        hub=EventHub()
        for i in range(1,4):hub.publish({'type':'caption','id':i,'expires_at':time.time()+3})
        self.assertEqual([r['id'] for r in hub.active],[2,3])


class OverlayHttpTests(unittest.TestCase):
    def test_static_whitelist_and_local_websocket_config(self):
        server=make_server(0,8770);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(base+'/config.json') as response:
                cfg=json.load(response);self.assertEqual(cfg['websocket'],'ws://127.0.0.1:8770');self.assertEqual(cfg['display_duration_s'],3)
            with urllib.request.urlopen(base+'/overlay') as response:
                self.assertIn("script-src 'self'",response.headers['Content-Security-Policy'])
                self.assertIn(b'Game sound captions',response.read())
            with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(base+'/.env')
            self.assertEqual(error.exception.code,404)
        finally:server.shutdown();server.server_close();thread.join()
