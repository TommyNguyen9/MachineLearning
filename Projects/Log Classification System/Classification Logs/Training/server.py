import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse

from classify import classify
from processor_bert import get_model
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield

app = FastAPI(lifespan = lifespan)

@app.post("/classify")
async def classify_logs(file: UploadFile):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code = 400, detail = "File must be a CSV.")
    
    try: 
        # READ CSV FILE:
        df = pd.read_csv(file.file)
        if "source" not in df.columns or "log_message" not in df.columns:
            raise HTTPException(status_code = 400, detail = "CSV must contain 'source and 'log_message columns.")
        
        # Classification:

        df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

        output_file = "../Resources/output.csv"
        df.to_csv(output_file, index = False)
        return FileResponse(output_file, media_type = 'text/csv')
    
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    

    
