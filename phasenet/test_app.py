import requests
import obspy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

### Start running the model first:
### FLASK_ENV=development FLASK_APP=app.py flask run

def read_data(mseed):
    data = []
    mseed = mseed.sort()
    for c in ["E", "N", "Z"]:
        data.append(mseed.select(channel="*"+c)[0].data)
    return np.array(data).T

timestamp = lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

## prepare some test data
mseed = obspy.read()
data = []
for i in range(1): 
    data.append(read_data(mseed))
data = {
    "id": ["test01"],
    "timestamp": [timestamp(datetime.now())],
    "vec": np.array(data).tolist(),
    "dt": 0.01
    }

## run prediction
print(data["id"])
resp = requests.get("http://localhost:8000/predict", json=data)
# picks = resp.json()["picks"]
print(resp.json())


## plot figure
plt.figure()
plt.plot(np.array(data["data"])[0,:,1])
ylim = plt.ylim()
plt.plot([picks[0][0][0], picks[0][0][0]], ylim, label="P-phase")
plt.text(picks[0][0][0], ylim[1]*0.9, f"{picks[0][1][0]:.2f}")
plt.plot([picks[0][2][0], picks[0][2][0]], ylim, label="S-phase")
plt.text(picks[0][2][0], ylim[1]*0.9, f"{picks[0][1][0]:.2f}")
plt.legend()
plt.savefig("test.png")