from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import time
import cv2
import numpy as np
import cvzone
from sse_starlette import EventSourceResponse

videoPath = 'videos/walking.mp4'
# videoPath = 0
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
modelPath = 'frozen_inference_graph.pb'
classesPath='coco.names'

app = FastAPI()

# Configurar permissões CORS
app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)

# Simulação de uma função de detecção de pessoas
def detect_person():
    net = cv2.dnn_DetectionModel(modelPath,configPath)

    net.setInputSize(320,320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5,127.5,127.5))
    net.setInputSwapRB(True)


    with open(classesPath,'r') as f:
        classesList = f.read().splitlines()

    video = cv2.VideoCapture(videoPath)

    while True:
        check,img = video.read()
        img = cv2.resize(img,(1270,720))

        labels, confs, bboxs = net.detect(img,confThreshold=0.5)

        bboxs = list(bboxs)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))

        bboxsIdx = cv2.dnn.NMSBoxes(bboxs,confs,score_threshold=0.5, nms_threshold=0.3)

        # INICIALIZA CONTADOR
        data_counter = ""
        counter = 0
        
        if len(bboxsIdx) !=0:
            for x in range(0,len(bboxsIdx)):
                bbox = bboxs[np.squeeze(bboxsIdx[x])]
                conf = confs[np.squeeze(bboxsIdx[x])]
                labelsID = np.squeeze(labels[np.squeeze(bboxsIdx[x])])-1
                label = classesList[labelsID]

                if labelsID == 0:
                    counter +=1
                    
                data_counter = f"{counter}\n"
                
                print(bbox,labelsID,conf)
                x,y,w,h = bbox

                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
                cvzone.putTextRect(img,f'{label} {round(conf,2)}',(x,y-10),colorR=(255,0,0),scale=1,thickness=2)

        cv2.imshow('Imagem',img)

        if cv2.waitKey(1)==27:
            break
        
        yield data_counter

@app.get("/events")
async def sse_endpoint():
    # Tentei com StreamingResponse, mas não estava dando certo
    return EventSourceResponse(detect_person(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
