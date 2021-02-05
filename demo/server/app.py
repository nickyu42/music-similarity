import os
import sys
import uuid
import time
import pickle
import pathlib
import tempfile
import threading
from multiprocessing import Process, Queue

from cachetools import TTLCache
from flask import Flask, Blueprint, jsonify, render_template, request, current_app, abort

from analysis.classification import MusicSimModel

CACHE_SIZE = 1024
CACHE_TTL = 120

api_routes = Blueprint('api_routes', __name__)


@api_routes.route('/status', methods=['POST'])
def status():
    if not request.is_json or 'id' not in request.json:
        abort(400, 'The request should be a of type "application/json" and should contain an "id" key')

    client_id = request.json['id']
    results_cache = current_app.config['result_cache']

    if client_id not in results_cache:
        return jsonify(done=False)

    success, label, probability = results_cache[client_id]

    return jsonify(done=True, success=success, label=label, probability=probability)


@api_routes.route('/song', methods=['POST'])
def song():
    song = request.files['song']

    # Generate random ID for tracking user requests
    client_id = uuid.uuid4().int

    queue = current_app.config['in_queue']
    queue.put(client_id)

    # XXX: Currently the file system is used as cache
    #   for something as small scale as this application
    #   it shouldn't be a problem
    file_cache = current_app.config['file_cache']
    with open((file_cache / str(client_id)).resolve(), 'wb') as f:
        song.save(f)

    return jsonify(id=client_id), 202


def load_model() -> MusicSimModel:
    if 'MODEL_PATH' not in os.environ:
        raise ValueError('Environment variable MODEL_PATH not set')

    model_path = pathlib.Path(os.environ.get('MODEL_PATH'))

    if not model_path.is_file():
        raise ValueError('MODEL_PATH does not exist or is not a file.')

    with model_path.open('rb') as file:
        return pickle.load(file)


def job_handler(model: MusicSimModel, in_queue: Queue, out_queue: Queue, file_cache: pathlib.Path):
    while True:
        try:
            client_id = in_queue.get()
        except:
            # XXX: Add proper handling
            continue

        # TODO: add model prediction

        out_queue.put(
            (client_id, {'success': True,
                         'label': 'shounen', probability: 100.0})
        )


def results_consumer(out_queue: Queue, results_cache: TTLCache, file_cache: pathlib.Path):
    while True:
        client_id, result = out_queue.get()
        results_cache[client_id] = result

        to_remove = (file_cache / str(client_id)).resolve()
        os.remove(to_remove)


def init_production(app: Flask):
    app.register_blueprint(api_routes)

    app.config['results_cache'] = TTLCache(CACHE_SIZE, CACHE_TTL)
    app.config['file_cache'] = pathlib.Path(tempfile.mkdtemp())

    print(f' * File cache created at {app.config["file_cache"]}')

    # TODO: add proper model
    # model = load_model()
    model = None

    in_queue = Queue()
    out_queue = Queue()

    # Add reference to input queue in the app's config
    app.config['in_queue'] = in_queue

    # Start background song predictor
    p = Process(
        target=job_handler,
        args=(model, in_queue, out_queue, app.config['file_cache']),
        daemon=True
    )
    p.start()
    print(' * Started song predictor process')

    t = threading.Thread(
        target=results_consumer,
        args=(
            out_queue,
            app.config['results_cache'],
            app.config['file_cache']
        ),
        daemon=True
    )
    t.start()
    print(' * Started cache cleaner thread')


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')

    if os.environ.get('FLASK_ENV') == 'production':
        init_production(app)

    return app
