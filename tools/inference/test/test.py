import requests



resp=requests.post("http://localhost:5000/predict")#,files={'file':open("/home/grajebhosle/Documents/IPML/Projects/Anamoly/anomalib-main/tools/inference/Data/2NOK_CROP/2.png",'rb')})


print(resp.text)
