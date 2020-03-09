import os
from logging.config import dictConfig

import yaml
from flask import Flask
from flask_jsonschema import JsonSchema
from config import BASE_DIR, Config, LOG_YAML
from pprint import pprint
from color_recognition.color_detection import ColorIdentify

__author__ = "zhouxing"

jsonschema = JsonSchema()

with open(LOG_YAML, 'rt') as f:
    config = yaml.safe_load(f.read())
    if "handlers" in config.keys():
        for k, v in config["handlers"].items():
            if "filename" in v.keys():
                v["filename"] = os.path.join(BASE_DIR, v["filename"])
    pprint(config)
dictConfig(config)


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    jsonschema.init_app(app)
    print("finished loaded app !")
    return app


app = create_app()

ci = ColorIdentify() # 服饰颜色鉴别器

# if __name__ == "__main__":
#     pass
