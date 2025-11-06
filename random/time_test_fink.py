import numpy as np
import requests
import io
import time

def get_lc(oid="ZTF25abyakba", columns="*"):
    r = requests.post(
        'https://api.fink-portal.org/api/v1/objects',
        json={
            'objectId': oid,
            "columns": columns,
            'output-format': 'json'
        }
    )

for oid in ["ZTF25abyakba", "ZTF17aaclogk"]:
    times = []
    for i in range(10):
        t0 = time.time()
        get_lc(oid=oid)
        times.append(time.time() - t0)
    print("oid: {}".format(oid))
    print("t = {:.2f} +/- {:.2f} seconds".format(np.mean(times), np.std(times)))