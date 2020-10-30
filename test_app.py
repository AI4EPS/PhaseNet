import requests
import obspy
import numpy as np
import matplotlib.pyplot as plt

### Start running the model first:
### FLASK_ENV=development FLASK_APP=app.py flask run

def read_data(mseed):
    data = []
    mseed = mseed.sort()
    for c in ["E", "N", "Z"]:
        data.append(mseed.select(channel="*"+c)[0].data)
    return np.array(data).T

## prepare some test data
mseed = obspy.read()
data = []
for i in range(10): 
    data.append(read_data(mseed))
data = {"data": np.array(data).tolist()}

## run prediction
resp = requests.post("http://localhost:5000/predict", json=data)
picks = resp.json()["picks"]


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