import json
import os
import socket
import traceback

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from oedisi.componentframework.system_configuration import ComponentStruct
from oedisi.types.common import BrokerConfig, DefaultFileNames, HeathCheck, ServerReply

from distopf_federate.federate import run_simulator

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
    response = HeathCheck(hostname=hostname, host_ip=host_ip).model_dump()
    return JSONResponse(response, 200)


@app.post("/run")
async def run_model(broker_config: BrokerConfig, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_simulator, broker_config)
        response = ServerReply(detail="Task successfully added.").model_dump()
        return JSONResponse(response, 200)
    except Exception:
        err = traceback.format_exc()
        raise HTTPException(500, str(err))


@app.post("/configure")
async def configure(component_struct: ComponentStruct):
    component = component_struct.component
    params = component.parameters
    params["name"] = component.name
    links = {}
    for link in component_struct.links:
        links[link.target_port] = f"{link.source}/{link.source_port}"
    with open(DefaultFileNames.INPUT_MAPPING.value, "w") as fh:
        json.dump(links, fh)
    with open(DefaultFileNames.STATIC_INPUTS.value, "w") as fh:
        json.dump(params, fh)
    response = ServerReply(
        detail="Successfully updated configuration files."
    ).model_dump()
    return JSONResponse(response, 200)


def main():
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PORT"]))


if __name__ == "__main__":
    main()
