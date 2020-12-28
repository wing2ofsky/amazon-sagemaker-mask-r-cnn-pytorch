# Add Chinese Character support since Python is encoded by ASCII by default
# -*- coding: UTF-8 -*-

import requests
import json

url='http://localhost:8080/invocations'

bucket = 'wzy-test-123'
image_uri = '2020.11.10处理/标的定损-0/现场查勘照片/05电池撞击故障灯 (1)11.jpg'
test_data = {
    'bucket' : bucket,
    'image_uri' : image_uri,
    'region' : 'cn-northwest-1',
    'content_type': "application/json",
}
payload = json.dumps(test_data)


r = requests.post(url,data=payload)

#show result
print (r.text)
