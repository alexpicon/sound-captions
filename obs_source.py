"""Read-only OBS WebSocket v5 adapter for one local Media Source's playhead."""
import base64
import hashlib
import json
import os
from urllib.parse import urlsplit
from websockets.asyncio.client import connect


def auth_response(password,salt,challenge):
    secret=base64.b64encode(hashlib.sha256((password+salt).encode()).digest()).decode()
    return base64.b64encode(hashlib.sha256((secret+challenge).encode()).digest()).decode()


class ObsClient:
    ALLOWED={'GetInputList','GetInputSettings','GetMediaInputStatus',
             'GetCurrentProgramScene','GetSceneItemList'}
    def __init__(self,url=None,password=None):
        self.url=url or os.environ.get('OBS_WS_URL','ws://127.0.0.1:4455')
        u=urlsplit(self.url)
        if u.scheme!='ws' or u.hostname not in {'127.0.0.1','localhost','::1'} or u.username or u.password:
            raise ValueError('OBS URL must be a local ws:// address; password belongs in OBS_WS_PASSWORD')
        self.password=password if password is not None else os.environ.get('OBS_WS_PASSWORD','')
        self.connection=None;self.sequence=0

    async def __aenter__(self):
        self.connection=await connect(self.url,proxy=None,open_timeout=3,close_timeout=1,max_size=1<<20)
        try:
            hello=json.loads(await self._recv())
            if hello.get('op')!=0:raise ValueError('Expected OBS Hello')
            data={'rpcVersion':1,'eventSubscriptions':0}
            challenge=hello['d'].get('authentication')
            if challenge:
                if not self.password:raise ValueError('Set OBS_WS_PASSWORD in .env to the OBS WebSocket password')
                data['authentication']=auth_response(self.password,challenge['salt'],challenge['challenge'])
            await self.connection.send(json.dumps({'op':1,'d':data}))
            if json.loads(await self._recv()).get('op')!=2:raise ValueError('OBS identification failed')
        except BaseException:
            await self.connection.close();raise
        return self

    async def _recv(self):
        import asyncio
        return await asyncio.wait_for(self.connection.recv(),3)

    async def __aexit__(self,*args):
        await self.connection.close()

    async def request(self,kind,data=None):
        if kind not in self.ALLOWED:raise ValueError('Only read-only OBS media queries are allowed')
        self.sequence+=1;request_id=str(self.sequence)
        await self.connection.send(json.dumps({'op':6,'d':{'requestType':kind,'requestId':request_id,'requestData':data or {}}}))
        response=json.loads(await self._recv())
        if response.get('op')!=7 or response['d'].get('requestId')!=request_id:
            raise ValueError('Unexpected OBS response')
        value=response['d']
        if not value['requestStatus']['result']:
            raise ValueError(f"OBS {kind} failed ({value['requestStatus'].get('code')})")
        return value.get('responseData',{})
