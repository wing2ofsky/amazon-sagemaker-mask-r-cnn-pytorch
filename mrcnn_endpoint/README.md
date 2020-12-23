# Maskrcnn-SageMaker
Deploy MaskRCNN (torch-vision)  on Amazon SageMaker


<a name="YOLOv4"></a>

## Features

- [x] **Use MaskRCNN pretrained model (coco) to deploy**


## Quick Start

---------------

# build the environment and test locally

server
~~~~shell script
sh build_and_push.sh
docker run -v -d -p 8080:8080 mrcnn-car
~~~~

client
~~~~shell script
import requests
import json

url='http://localhost:8080/invocations'

bucket = 'spot-bot-asset'
image_uri = 'end/car.jpg'
test_data = {
    'bucket' : bucket,
    'image_uri' : image_uri,
    'content_type': "application/json",
}
payload = json.dumps(test_data)


r = requests.post(url,data=payload)

#show result
print (r.text)
CPU times: user 5.8 ms, sys: 0 ns, total: 5.8 ms
Wall time: 4.05 s
~~~~

result
~~~~
{"result": {"label": "withcar"}}
~~~~

# build endpoint
~~~~
cd mrcnn_endpoint
python create_endpoint.py
~~~~

# use endpoint
~~~~ python
def infer(input_image):
    from boto3.session import Session
    import json

    bucket = 'lianbao-mask-rcnn'
    image_uri = input_image
    test_data = {
        'bucket' : bucket,
        'image_uri' : image_uri,
        'content_type': "application/json",
    }
    payload = json.dumps(test_data)


    session = Session()

    runtime = session.client("runtime.sagemaker")
    response = runtime.invoke_endpoint(
        EndpointName='mrcnn-car',
        ContentType="application/json",
        Body=payload)

    result = json.loads(response["Body"].read())
    print (result)
    
infer('test.jpg')
~~~~

multi-process

~~~~ python
%%time 
import multiprocessing as mul

pool = mul.Pool(5)
rel = pool.map(infer, ['end/test2.jpg','end/test.jpg','end/test2.jpg','end/test2.jpg','end/test2.jpg','end/test2.jpg'])
print(rel)
~~~~

