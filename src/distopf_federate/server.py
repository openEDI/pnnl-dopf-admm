import json
import logging
import os
import socket
import traceback

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from oedisi.types.common import BrokerConfig, DefaultFileNames, HeathCheck, ServerReply
from pydantic import ValidationError

from distopf_federate.federate import run_simulator
from distopf_federate.schemas import DynamicInputs, DynamicOutputs, StaticInputs

logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = "127.0.0.1"
    try:
        host_ip = socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        try:
            host_ip = socket.gethostbyname(socket.gethostname() + ".local")
        except socket.gaierror:
            pass
    return JSONResponse(
        HeathCheck(hostname=hostname, host_ip=host_ip).model_dump(), 200
    )


@app.post("/configure")
async def configure(config: dict):
    try:
        raw_static = config.get("static_inputs", {})
        raw_static["name"] = config.get("name", raw_static.get("name", ""))
        validated_static = StaticInputs.model_validate(raw_static)

        links: dict[str, str] = {}
        for link in config.get("links", []):
            links[link["target_port"]] = f"{link['source']}/{link['source_port']}"

        validated_dyn_inputs = DynamicInputs.model_validate(links)
        validated_dyn_outputs = DynamicOutputs.model_validate(links)

        with open(DefaultFileNames.INPUT_MAPPING.value, "w") as fh:
            json.dump(links, fh, indent=2)
        with open(DefaultFileNames.STATIC_INPUTS.value, "w") as fh:
            json.dump(validated_static.model_dump(), fh, indent=2)

        return JSONResponse(
            ServerReply(detail="Successfully updated configuration files.").model_dump(),
            200,
        )
    except ValidationError as exc:
        logger.error("Configuration validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {exc}")
    except Exception:
        err = traceback.format_exc()
        logger.error("Configuration failed: %s", err)
        raise HTTPException(status_code=500, detail=err)



@app.post("/run")
async def run_model(broker_config: BrokerConfig, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_simulator, broker_config)
        return JSONResponse(
            ServerReply(detail="Task successfully added.").model_dump(), 200
        )
    except Exception:
        err = traceback.format_exc()
        raise HTTPException(500, str(err))


def main():
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PORT"]))


if __name__ == "__main__":
    main()
